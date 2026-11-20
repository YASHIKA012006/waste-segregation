import os
import json
import sqlite3
from datetime import date, datetime
from functools import wraps

import numpy as np
from flask import Flask, flash, g, redirect, render_template, request, session, url_for
from PIL import Image, UnidentifiedImageError
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATABASE = os.path.join(BASE_DIR, 'database', 'waste.db')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'waste_model.keras')
CLASS_NAMES_PATH = os.path.join(BASE_DIR, 'model', 'class_names.json')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
DEFAULT_CATEGORIES = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def discover_categories():
    if os.path.isfile(CLASS_NAMES_PATH):
        try:
            with open(CLASS_NAMES_PATH, encoding='utf-8') as mapping:
                names = json.load(mapping)
            if isinstance(names, list) and names:
                return names
        except (OSError, ValueError):
            pass
    for current, directories, files in os.walk(DATASET_DIR):
        found = []
        for name in directories:
            folder = os.path.join(current, name)
            if any(os.path.splitext(item)[1].lower() in IMAGE_EXTENSIONS for item in os.listdir(folder)):
                found.append(name)
        if len(found) >= 2:
            return sorted(found)
    return DEFAULT_CATEGORIES


CATEGORIES = discover_categories()
RECOMMENDATIONS = {
    'plastic': 'Clean recyclable plastic and place it with dry/recyclable waste.',
    'paper': 'Keep paper dry and send it for paper recycling.',
    'cardboard': 'Flatten cardboard and send it for recycling.',
    'biological': 'Dispose through wet/organic waste collection or composting where available.',
    'metal': 'Separate metal items and send them for metal recycling.',
    'green glass': 'Handle glass carefully and send it for glass recycling.',
    'brown glass': 'Handle glass carefully and send it for glass recycling.',
    'white glass': 'Handle glass carefully and send it for glass recycling.',
    'glass': 'Handle glass carefully and send it for glass recycling.',
    'clothes': 'Donate reusable clothes or send damaged textiles for textile recycling.',
    'shoes': 'Donate usable shoes or use appropriate textile/shoe recycling.',
    'battery': 'Do not throw batteries into regular household waste. Use an appropriate battery/e-waste collection point.',
    'trash': 'Dispose according to local waste collection guidelines.',
}

RECYCLABLE_CATEGORIES = {
    'paper', 'cardboard', 'plastic', 'metal', 'green glass', 'brown glass',
    'white glass', 'clothes', 'shoes', 'battery', 'batteries', 'glass'
}
ECO_POINTS = {
    'paper': 3, 'cardboard': 3, 'plastic': 5, 'metal': 7,
    'green glass': 5, 'brown glass': 5, 'white glass': 5, 'glass': 5,
    'battery': 10, 'batteries': 10, 'biological': 2,
    'clothes': 4, 'shoes': 4, 'trash': 0,
}


def recommendation_for(category):
    key = category.strip().replace('-', ' ').replace('_', ' ').lower()
    return RECOMMENDATIONS.get(key, f'Sort {category} separately and follow your local waste collection guidelines.')


def recyclability_for(category):
    key = category.strip().replace('-', ' ').replace('_', ' ').lower()
    return 'RECYCLABLE' if key in RECYCLABLE_CATEGORIES else 'NON-RECYCLABLE'


def points_for(category, verified_recyclability):
    key = category.strip().replace('-', ' ').replace('_', ' ').lower()
    # Organic/biological waste still earns its fixed composting points even
    # though it is suggested as non-recyclable in the material sort.
    if verified_recyclability != 'RECYCLABLE' and key != 'biological':
        return 0
    return ECO_POINTS.get(key, 0)
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'local-development-key-change-me')
app.config.update(UPLOAD_FOLDER=UPLOAD_FOLDER, MAX_CONTENT_LENGTH=8 * 1024 * 1024)
os.makedirs(os.path.dirname(DATABASE), exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

MODEL = None
MODEL_LOAD_ERROR = None
if os.path.exists(MODEL_PATH):
    try:
        from tensorflow.keras.models import load_model
        MODEL = load_model(MODEL_PATH)
    except Exception as exc:
        MODEL_LOAD_ERROR = str(exc)


def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
        g.db.execute('PRAGMA foreign_keys = ON')
    return g.db


@app.teardown_appcontext
def close_db(_error=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DATABASE)
    db.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE, password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user', created_at TEXT NOT NULL,
            eco_points INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS waste_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id), image_filename TEXT NOT NULL,
            category TEXT NOT NULL, confidence REAL NOT NULL, recommendation TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS collection_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL REFERENCES users(id),
            waste_id INTEGER REFERENCES waste_records(id), category TEXT NOT NULL,
            quantity TEXT NOT NULL, address TEXT NOT NULL, phone TEXT NOT NULL,
            preferred_date TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING' CHECK(status IN ('PENDING','ACCEPTED','REJECTED','COLLECTED','COMPLETED')),
            created_at TEXT NOT NULL,
            suggested_recyclability TEXT NOT NULL DEFAULT 'NON-RECYCLABLE',
            verified_recyclability TEXT,
            points_awarded INTEGER NOT NULL DEFAULT 0,
            points_value INTEGER NOT NULL DEFAULT 0,
            verified_at TEXT
        );
    ''')
    user_columns = {row[1] for row in db.execute('PRAGMA table_info(users)')}
    if 'eco_points' not in user_columns:
        db.execute('ALTER TABLE users ADD COLUMN eco_points INTEGER NOT NULL DEFAULT 0')
    request_columns = {row[1] for row in db.execute('PRAGMA table_info(collection_requests)')}
    additions = {
        'suggested_recyclability': "TEXT NOT NULL DEFAULT 'NON-RECYCLABLE'",
        'verified_recyclability': 'TEXT',
        'points_awarded': 'INTEGER NOT NULL DEFAULT 0',
        'points_value': 'INTEGER NOT NULL DEFAULT 0',
        'verified_at': 'TEXT',
    }
    for column, declaration in additions.items():
        if column not in request_columns:
            db.execute(f'ALTER TABLE collection_requests ADD COLUMN {column} {declaration}')
    for record_id, category in db.execute('SELECT id,category FROM collection_requests WHERE suggested_recyclability IS NULL OR suggested_recyclability="NON-RECYCLABLE"'):
        suggested = recyclability_for(category)
        db.execute('UPDATE collection_requests SET suggested_recyclability=? WHERE id=?', (suggested, record_id))
    if not db.execute("SELECT id FROM users WHERE email = ?", ('admin@example.com',)).fetchone():
        db.execute('INSERT INTO users(name,email,password,role,created_at) VALUES(?,?,?,?,?)',
                   ('Development Admin', 'admin@example.com', generate_password_hash('admin123'), 'admin', datetime.now().isoformat(timespec='seconds')))
    db.commit()
    db.close()


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get('user_id'):
            flash('Please log in to continue.', 'error')
            return redirect(url_for('login'))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get('user_id'):
            flash('Please log in as an administrator.', 'error')
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            flash('You do not have permission to view that page.', 'error')
            return redirect(url_for('dashboard'))
        return view(*args, **kwargs)
    return wrapped


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def classify(image_path):
    if MODEL is None:
        if MODEL_LOAD_ERROR:
            raise RuntimeError('The trained model could not be loaded. Reinstall its TensorFlow dependency and retrain the model.')
        raise RuntimeError('The waste classification model has not been trained yet. Follow the dataset and training steps in README.md.')
    try:
        image = Image.open(image_path).convert('RGB').resize((224, 224))
        pixels = np.asarray(image, dtype=np.float32)
        pixels = np.expand_dims(pixels, axis=0)
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        probabilities = MODEL.predict(preprocess_input(pixels), verbose=0)[0]
        if len(probabilities) != len(CATEGORIES):
            raise ValueError('Model output does not match model/class_names.json.')
        index = int(np.argmax(probabilities))
        return CATEGORIES[index], float(probabilities[index])
    except Exception as exc:
        raise RuntimeError('The model could not analyze this image. Check the trained model and try another image.') from exc


@app.context_processor
def inject_globals():
    values = {'categories': CATEGORIES, 'current_year': datetime.now().year,
              'recyclability_for': recyclability_for}
    if session.get('role') == 'admin':
        values['admin_pending_count'] = get_db().execute(
            "SELECT COUNT(*) FROM collection_requests WHERE status='PENDING'"
        ).fetchone()[0]
    return values


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name, email, password = request.form.get('name', '').strip(), request.form.get('email', '').strip().lower(), request.form.get('password', '')
        if not name or not email or len(password) < 8:
            flash('Enter your name and email, and use a password with at least 8 characters.', 'error')
        else:
            try:
                db = get_db()
                db.execute('INSERT INTO users(name,email,password,role,created_at) VALUES(?,?,?,?,?)',
                           (name, email, generate_password_hash(password), 'user', datetime.now().isoformat(timespec='seconds')))
                db.commit()
                flash('Account created. You can now log in.', 'success')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                flash('An account with that email already exists.', 'error')
            except sqlite3.Error:
                flash('We could not save your account. Please try again.', 'error')
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('role') == 'admin':
        return redirect(url_for('admin_dashboard'))
    if session.get('user_id'):
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        user = get_db().execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        if user and user['role'] == 'user' and check_password_hash(user['password'], request.form.get('password', '')):
            session.clear()
            session['user_id'], session['name'], session['role'] = user['id'], user['name'], user['role']
            return redirect(url_for('dashboard'))
        flash('Email or password is incorrect.', 'error')
    return render_template('login.html')


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if session.get('role') == 'admin':
        return redirect(url_for('admin_dashboard'))
    if session.get('user_id'):
        flash('Please log out of the user account before using Admin Login.', 'error')
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        user = get_db().execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        if user and user['role'] == 'admin' and check_password_hash(user['password'], request.form.get('password', '')):
            session.clear()
            session['user_id'], session['name'], session['role'] = user['id'], user['name'], user['role']
            return redirect(url_for('admin_dashboard'))
        flash('Administrator email or password is incorrect.', 'error')
    return render_template('admin_login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('index'))


@app.route('/identify', methods=['GET', 'POST'])
@login_required
def identify():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or not file.filename:
            flash('Choose an image before analyzing.', 'error')
            return redirect(url_for('identify'))
        if not allowed_file(file.filename):
            flash('Please upload a JPG, JPEG, PNG, or WEBP image.', 'error')
            return redirect(url_for('identify'))
        filename = secure_filename(file.filename)
        unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{filename}"
        path = os.path.join(UPLOAD_FOLDER, unique_filename)
        try:
            file.save(path)
            with Image.open(path) as img:
                img.verify()
            category, confidence = classify(path)
            db = get_db()
            cursor = db.execute('INSERT INTO waste_records(user_id,image_filename,category,confidence,recommendation,created_at) VALUES(?,?,?,?,?,?)',
                                (session['user_id'], unique_filename, category, confidence, recommendation_for(category), datetime.now().isoformat(timespec='seconds')))
            db.commit()
            record = db.execute('SELECT * FROM waste_records WHERE id=?', (cursor.lastrowid,)).fetchone()
            return render_template('result.html', record=record)
        except UnidentifiedImageError:
            if os.path.exists(path): os.remove(path)
            flash('That image could not be opened. Try a different image file.', 'error')
        except RuntimeError as exc:
            if os.path.exists(path): os.remove(path)
            flash(str(exc), 'error')
        except (OSError, sqlite3.Error):
            if os.path.exists(path): os.remove(path)
            flash('We could not process that image. Please try again.', 'error')
        return redirect(url_for('identify'))
    return render_template('identify.html')


@app.route('/dashboard')
@login_required
def dashboard():
    db = get_db()
    uid = session['user_id']
    user_points = db.execute('SELECT eco_points FROM users WHERE id=?', (uid,)).fetchone()['eco_points']
    stats = {
        'identified': db.execute('SELECT COUNT(*) FROM waste_records WHERE user_id=?', (uid,)).fetchone()[0],
        'requests': db.execute('SELECT COUNT(*) FROM collection_requests WHERE user_id=?', (uid,)).fetchone()[0],
        'pending': db.execute("SELECT COUNT(*) FROM collection_requests WHERE user_id=? AND status='PENDING'", (uid,)).fetchone()[0],
        'completed': db.execute("SELECT COUNT(*) FROM collection_requests WHERE user_id=? AND status='COMPLETED'", (uid,)).fetchone()[0],
        'collected_items': db.execute("SELECT COUNT(*) FROM collection_requests WHERE user_id=? AND status IN ('COLLECTED','COMPLETED')", (uid,)).fetchone()[0],
        'recyclable_items': db.execute("SELECT COUNT(*) FROM collection_requests WHERE user_id=? AND verified_recyclability='RECYCLABLE'", (uid,)).fetchone()[0],
    }
    records = db.execute('SELECT * FROM collection_requests WHERE user_id=? ORDER BY id DESC', (uid,)).fetchall()
    history = db.execute('''SELECT category,points_value,verified_at FROM collection_requests
        WHERE user_id=? AND points_awarded=1 ORDER BY verified_at DESC,id DESC''', (uid,)).fetchall()
    badge = 'Eco Champion' if user_points >= 100 else 'Green Recycler' if user_points >= 50 else 'Eco Starter' if user_points >= 25 else 'No badge yet'
    return render_template('dashboard.html', stats=stats, records=records,
                           eco_points=user_points, points_history=history, eco_badge=badge)


@app.route('/collection', methods=['GET', 'POST'])
@login_required
def collection():
    db = get_db()
    waste_id = request.values.get('waste_id', type=int)
    waste = db.execute('SELECT * FROM waste_records WHERE id=? AND user_id=?', (waste_id, session['user_id'])).fetchone() if waste_id else None
    if not waste:
        flash('Identify an item first, then request collection from its result.', 'error')
        return redirect(url_for('identify'))
    if request.method == 'POST':
        category = waste['category']
        quantity = request.form.get('quantity', '').strip()
        address = request.form.get('address', '').strip()
        phone = request.form.get('phone', '').strip()
        preferred_date = request.form.get('preferred_date', '')
        try:
            amount = float(quantity)
            valid = 0 < amount <= 10000 and bool(address) and len(address) <= 500 and phone.replace('+', '').replace(' ', '').replace('-', '').isdigit() and 7 <= len(phone.replace('+', '').replace(' ', '').replace('-', '')) <= 15 and date.fromisoformat(preferred_date) >= date.today()
        except (ValueError, TypeError):
            valid = False
        if not valid:
            flash('Check the category, positive quantity, address, phone number, and a date today or later.', 'error')
        else:
            try:
                db.execute('''INSERT INTO collection_requests(user_id,waste_id,category,quantity,address,phone,preferred_date,status,created_at,suggested_recyclability)
                    VALUES(?,?,?,?,?,?,?,?,?,?)''',
                           (session['user_id'], waste['id'], category, quantity, address, phone, preferred_date, 'PENDING', datetime.now().isoformat(timespec='seconds'), recyclability_for(category)))
                db.commit()
                flash('Your collection request has been submitted.', 'success')
                return redirect(url_for('dashboard'))
            except sqlite3.Error:
                flash('We could not save your request. Please try again.', 'error')
    return render_template('collection.html', waste=waste, today=date.today().isoformat())


@app.route('/admin')
@admin_required
def admin_dashboard():
    db = get_db()
    stats = {
        'users': db.execute("SELECT COUNT(*) FROM users WHERE role='user'").fetchone()[0],
        'waste': db.execute('SELECT COUNT(*) FROM waste_records').fetchone()[0],
        'total_requests': db.execute('SELECT COUNT(*) FROM collection_requests').fetchone()[0],
        'pending': db.execute("SELECT COUNT(*) FROM collection_requests WHERE status='PENDING'").fetchone()[0],
        'accepted': db.execute("SELECT COUNT(*) FROM collection_requests WHERE status='ACCEPTED'").fetchone()[0],
        'collected': db.execute("SELECT COUNT(*) FROM collection_requests WHERE status='COLLECTED'").fetchone()[0],
        'completed': db.execute("SELECT COUNT(*) FROM collection_requests WHERE status='COMPLETED'").fetchone()[0],
        'rejected': db.execute("SELECT COUNT(*) FROM collection_requests WHERE status='REJECTED'").fetchone()[0],
        'eco_points': db.execute('SELECT COALESCE(SUM(points_value),0) FROM collection_requests WHERE points_awarded=1').fetchone()[0],
    }
    category_rows = db.execute('SELECT category,COUNT(*) AS total FROM waste_records GROUP BY category ORDER BY category COLLATE NOCASE').fetchall()
    requests = db.execute('''SELECT r.*,u.name AS user_name,u.email AS user_email,w.confidence AS ai_confidence
        FROM collection_requests r JOIN users u ON u.id=r.user_id
        LEFT JOIN waste_records w ON w.id=r.waste_id
        ORDER BY CASE r.status WHEN 'PENDING' THEN 0 ELSE 1 END,r.id DESC LIMIT 6''').fetchall()
    return render_template('admin.html', stats=stats,
                           chart_labels=[row['category'] for row in category_rows],
                           chart_counts=[row['total'] for row in category_rows],
                           requests=requests)


@app.route('/admin/requests')
@admin_required
def admin_requests():
    requested_status = request.args.get('status', 'ALL').strip().upper()
    status = requested_status
    if status not in {'ALL', 'PENDING', 'ACCEPTED', 'COLLECTED', 'COMPLETED', 'REJECTED'}:
        status = 'ALL'
    search = request.args.get('q', '').strip()[:100]
    canonical_args = {}
    if status != 'ALL':
        canonical_args['status'] = status
    if search:
        canonical_args['q'] = search
    if request.args.to_dict(flat=True) != canonical_args:
        return redirect(url_for('admin_requests', **canonical_args))
    pattern = f'%{search}%'
    records = get_db().execute('''SELECT r.*,u.name AS user_name,u.email AS user_email,
          w.confidence AS ai_confidence
        FROM collection_requests r
        JOIN users u ON u.id=r.user_id
        LEFT JOIN waste_records w ON w.id=r.waste_id
        WHERE (?='ALL' OR r.status=?)
          AND (?='' OR u.name LIKE ? OR u.email LIKE ? OR r.category LIKE ? OR CAST(r.id AS TEXT) LIKE ?)
        ORDER BY CASE r.status WHEN 'PENDING' THEN 0 ELSE 1 END,r.id DESC''',
        (status, status, search, pattern, pattern, pattern, pattern)).fetchall()
    return render_template('admin_requests.html', records=records, active_status=status, search=search)


@app.route('/admin/waste-records')
@admin_required
def admin_waste_records():
    records = get_db().execute('''SELECT w.*,u.name AS user_name FROM waste_records w
        JOIN users u ON u.id=w.user_id ORDER BY w.id DESC''').fetchall()
    return render_template('admin_waste_records.html', records=records)


@app.route('/admin/users')
@admin_required
def admin_users():
    users = get_db().execute("SELECT id,name,email,created_at FROM users WHERE role='user' ORDER BY id DESC").fetchall()
    return render_template('admin_users.html', users=users)


@app.route('/admin/requests/<int:request_id>/status', methods=['POST'])
@admin_required
def update_status(request_id):
    status = request.form.get('status', '')
    transitions = {'PENDING': {'ACCEPTED', 'REJECTED'}, 'ACCEPTED': {'COLLECTED'}}
    db = get_db()
    row = db.execute('SELECT status,verified_recyclability,points_awarded FROM collection_requests WHERE id=?', (request_id,)).fetchone()
    allowed = bool(row and status in transitions.get(row['status'], set()))
    if status == 'COMPLETED':
        allowed = bool(row and row['status'] == 'COLLECTED' and row['verified_recyclability'] and row['points_awarded'])
    if allowed:
        db.execute('UPDATE collection_requests SET status=? WHERE id=?', (status, request_id))
        db.commit()
        flash(f'Request status updated to {status.lower()}.', 'success')
    else:
        flash('That status update is not available.', 'error')
    return redirect(url_for('admin_requests'))


@app.route('/admin/requests/<int:request_id>/verify', methods=['POST'])
@admin_required
def verify_recyclability(request_id):
    verified = request.form.get('verified_recyclability', '')
    if verified not in {'RECYCLABLE', 'NON-RECYCLABLE'}:
        flash('Choose Recyclable or Non-Recyclable.', 'error')
        return redirect(url_for('admin_requests'))
    db = get_db()
    try:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('''SELECT user_id,category,status,verified_recyclability,points_awarded
            FROM collection_requests WHERE id=?''', (request_id,)).fetchone()
        if not row or row['status'] not in {'COLLECTED', 'COMPLETED'} or row['verified_recyclability'] is not None or row['points_awarded']:
            db.rollback()
            flash('Recyclability can only be verified once after collection.', 'error')
            return redirect(url_for('admin_requests'))
        points = points_for(row['category'], verified)
        now = datetime.now().isoformat(timespec='seconds')
        updated = db.execute('''UPDATE collection_requests
            SET verified_recyclability=?,points_awarded=1,points_value=?,verified_at=?
            WHERE id=? AND status IN ('COLLECTED','COMPLETED') AND verified_recyclability IS NULL AND points_awarded=0''',
            (verified, points, now, request_id)).rowcount
        if updated != 1:
            db.rollback()
            flash('This request has already been verified.', 'error')
            return redirect(url_for('admin_requests'))
        db.execute('UPDATE users SET eco_points=eco_points+? WHERE id=?', (points, row['user_id']))
        db.commit()
        flash(f'Recyclability verified. {points} Eco Points awarded.', 'success')
    except sqlite3.Error:
        db.rollback()
        flash('We could not save recyclability verification.', 'error')
    return redirect(url_for('admin_requests'))


@app.errorhandler(413)
def file_too_large(_error):
    flash('Image is too large. Maximum upload size is 8 MB.', 'error')
    return redirect(url_for('identify'))


@app.errorhandler(500)
def server_error(_error):
    return render_template('error.html'), 500


init_db()

if __name__ == '__main__':
    app.run(debug=os.environ.get('FLASK_DEBUG') == '1')
