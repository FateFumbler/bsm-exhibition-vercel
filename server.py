"""
BSM Exhibition Backend - Flask Server (Vercel-ready version)
Handles API requests, file uploads, OCR processing, and SQLite database
Optimized for Vercel's serverless environment
"""
import os
import sqlite3
import uuid
import json
import base64
import time
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import openai

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) or '/home/Fate/.openclaw/workspace/bsm-exhibition-vercel'
DATABASE_PATH = os.path.join(BASE_DIR, 'database.db')

# OpenAI Configuration
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY environment variable is not set")

# Initialize OpenAI client
if OPENAI_API_KEY:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Flask app setup
app = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'static'), static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
CORS(app)

# Database initialization
def init_database():
    """Initialize SQLite database with contacts and industries tables"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Contacts table - images stored as base64 JSON
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS contacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT,
            company TEXT,
            designation TEXT,
            phone TEXT,
            email TEXT,
            website TEXT,
            industry TEXT,
            images TEXT,  -- JSON array of base64 images
            created_at TEXT
        )
    ''')
    
    # Industries table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS industries (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            is_default INTEGER DEFAULT 0
        )
    ''')
    
    # Insert default industries if not exist
    default_industries = [
        ('it', 'IT', 1),
        ('finance', 'Finance', 1),
        ('healthcare', 'Healthcare', 1),
        ('manufacturing', 'Manufacturing', 1),
        ('retail', 'Retail', 1),
        ('realestate', 'Real Estate', 1),
        ('education', 'Education', 1),
        ('marketing', 'Marketing', 1),
        ('consulting', 'Consulting', 1),
        ('other', 'Other (Type Your Own)', 1)
    ]
    
    cursor.executemany(
        'INSERT OR IGNORE INTO industries (id, name, is_default) VALUES (?, ?, ?)',
        default_industries
    )
    
    conn.commit()
    conn.close()
    print(f"Database initialized at: {DATABASE_PATH}")

# Database helpers
def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def dict_from_row(row):
    """Convert sqlite3.Row to dict"""
    return dict(row) if row else None

# GPT-4o Vision OCR function (synchronous for Vercel)
def extract_with_gpt4v(image_base64):
    """
    Extract contact info from business card image using GPT-4o Vision.
    Falls back to regex parsing if API fails.
    """
    if not OPENAI_API_KEY or not openai_client:
        raise Exception("OPENAI_API_KEY not configured")
    
    try:
        # Prepare the image for OpenAI
        if ',' in image_base64:
            # Remove data URL prefix if present
            image_base64 = image_base64.split(',')[1]
        
        # Call OpenAI GPT-4o with vision
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all contact information from this business card. Return ONLY valid JSON with these exact fields: fullName, company, designation, phone, email, website. Do not include any explanation or markdown formatting."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.1
        )
        
        # Parse the response
        content = response.choices[0].message.content.strip()
        
        # Try to extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            extracted_data = json.loads(json_match.group())
            
            # Normalize field names
            return {
                'fullName': extracted_data.get('fullName', extracted_data.get('name', '')),
                'company': extracted_data.get('company', ''),
                'designation': extracted_data.get('designation', extracted_data.get('title', '')),
                'phone': extracted_data.get('phone', ''),
                'email': extracted_data.get('email', ''),
                'website': extracted_data.get('website', '')
            }
        else:
            raise Exception("Could not parse JSON from GPT-4o response")
            
    except Exception as e:
        print(f"GPT-4o Vision OCR failed: {e}")
        raise

def parse_ocr_text(text):
    """Parse text to extract contact information (fallback)"""
    import re
    
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    
    result = {
        'fullName': '',
        'company': '',
        'designation': '',
        'phone': '',
        'email': '',
        'website': ''
    }
    
    # Email pattern
    email_regex = r'[\w.-]+@[\w.-]+\.\w+'
    emails = re.findall(email_regex, text)
    if emails:
        result['email'] = emails[0]
    
    # Website pattern
    website_regex = r'(?:https?://)?(?:www\.)?[\w-]+\.(?:com|org|net|io|co|biz|info|me|us|uk|in|au|ca|xyz|tech|app|dev|live|online|site|web|store|shop)(?:/[\w\-./?%&=]*)?'
    websites = re.findall(website_regex, text, re.IGNORECASE)
    if websites:
        website = websites[0]
        if not website.startswith('http'):
            website = 'https://' + website
        result['website'] = website
    
    # Phone pattern
    phone_regex = r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}'
    phones = re.findall(phone_regex, text)
    if phones:
        valid_phone = next((p for p in phones if len(re.sub(r'\D', '', p)) >= 8), None)
        if valid_phone:
            result['phone'] = valid_phone
    
    # Company keywords
    company_keywords = ['inc', 'corp', 'llc', 'ltd', 'co.', 'company', 'group', 'solutions', 
                        'services', 'technologies', 'tech', 'systems', 'enterprises', 'pvt', 'private']
    
    # Title keywords
    title_keywords = ['ceo', 'cto', 'cfo', 'director', 'manager', 'president', 'vp', 'head', 
                     'lead', 'engineer', 'developer', 'designer', 'consultant', 'analyst', 
                     'executive', 'officer', 'chief', 'senior', 'junior', 'associate', 
                     'coordinator', 'specialist', 'representative', 'sales', 'marketing', 
                     'account', 'partner', 'founder', 'owner']
    
    for line in lines:
        lower_line = line.lower()
        
        # Extract company
        if not result['company']:
            for keyword in company_keywords:
                if keyword in lower_line:
                    result['company'] = line
                    break
        
        # Extract designation
        if not result['designation']:
            for keyword in title_keywords:
                if keyword in lower_line:
                    result['designation'] = line
                    break
        
        # Extract name (simple heuristic)
        if not result['fullName']:
            if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}$', line):
                if not any(k in lower_line for k in company_keywords + title_keywords):
                    if len(line.split()) <= 4:
                        result['fullName'] = line
    
    return result

# Synchronous OCR processing for Vercel
def process_ocr_sync(base64_data):
    """Process OCR synchronously - suitable for Vercel serverless"""
    try:
        if not OPENAI_API_KEY:
            return {
                'fullName': '',
                'company': '',
                'designation': '',
                'phone': '',
                'email': '',
                'website': '',
                '_warning': 'OPENAI_API_KEY not configured'
            }
        
        # Try GPT-4o Vision first
        try:
            extracted_data = extract_with_gpt4v(base64_data)
            return extracted_data
        except Exception as gpt_error:
            print(f"GPT-4o Vision failed: {gpt_error}")
            return {
                'fullName': '',
                'company': '',
                'designation': '',
                'phone': '',
                'email': '',
                'website': '',
                '_fallback_note': 'GPT-4o Vision failed. Install easyocr for fallback.'
            }
        
    except Exception as e:
        return {'error': str(e)}

# API Routes
@app.route('/')
def index():
    """Serve the main frontend"""
    return send_from_directory(os.path.join(BASE_DIR, 'static'), 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory(os.path.join(BASE_DIR, 'static'), filename)

# Contacts API
@app.route('/api/contacts', methods=['GET'])
def get_contacts():
    """Get all contacts"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM contacts ORDER BY created_at DESC')
    rows = cursor.fetchall()
    conn.close()
    
    contacts = []
    for row in rows:
        contact = dict(row)
        # Parse images JSON
        if contact.get('images'):
            try:
                contact['images'] = json.loads(contact['images'])
            except:
                contact['images'] = []
        else:
            contact['images'] = []
        contacts.append(contact)
    
    return jsonify(contacts)

@app.route('/api/contacts', methods=['POST'])
def create_contact():
    """Create a new contact"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Validate at least name or company exists
    if not data.get('fullName') and not data.get('company'):
        return jsonify({'error': 'Name or company is required'}), 400
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Handle images - store as JSON (base64 in SQLite for Vercel)
    images_json = json.dumps(data.get('images', []))
    
    cursor.execute('''
        INSERT INTO contacts (full_name, company, designation, phone, email, website, industry, images, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data.get('fullName'),
        data.get('company'),
        data.get('designation'),
        data.get('phone'),
        data.get('email'),
        data.get('website'),
        data.get('industry'),
        images_json,
        datetime.now().isoformat()
    ))
    
    contact_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return jsonify({'id': contact_id, 'message': 'Contact created successfully'}), 201

@app.route('/api/contacts/<int:contact_id>', methods=['DELETE'])
def delete_contact(contact_id):
    """Delete a contact"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if contact exists
    cursor.execute('SELECT id FROM contacts WHERE id = ?', (contact_id,))
    if not cursor.fetchone():
        conn.close()
        return jsonify({'error': 'Contact not found'}), 404
    
    cursor.execute('DELETE FROM contacts WHERE id = ?', (contact_id,))
    conn.commit()
    conn.close()
    
    return jsonify({'message': 'Contact deleted successfully'})

@app.route('/api/contacts/clear', methods=['DELETE'])
def clear_all_contacts():
    """Clear all contacts"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM contacts')
    conn.commit()
    conn.close()
    
    return jsonify({'message': 'All contacts cleared'})

# Industries API
@app.route('/api/industries', methods=['GET'])
def get_industries():
    """Get all industries"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM industries ORDER BY is_default DESC, name ASC')
    rows = cursor.fetchall()
    conn.close()
    
    industries = [dict(row) for row in rows]
    return jsonify(industries)

@app.route('/api/industries', methods=['POST'])
def create_industry():
    """Create a new industry"""
    data = request.get_json()
    
    if not data or not data.get('name'):
        return jsonify({'error': 'Industry name is required'}), 400
    
    industry_id = data.get('id', data['name'].lower().replace(' ', '-') + '_' + str(int(time.time())))
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            'INSERT INTO industries (id, name, is_default) VALUES (?, ?, 0)',
            (industry_id, data['name'])
        )
        conn.commit()
        conn.close()
        
        return jsonify({'id': industry_id, 'message': 'Industry created successfully'}), 201
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'error': 'Industry already exists'}), 400

@app.route('/api/industries/<industry_id>', methods=['PUT'])
def update_industry(industry_id):
    """Update an industry"""
    data = request.get_json()
    
    if not data or not data.get('name'):
        return jsonify({'error': 'Industry name is required'}), 400
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('UPDATE industries SET name = ? WHERE id = ?', (data['name'], industry_id))
    
    if cursor.rowcount == 0:
        conn.close()
        return jsonify({'error': 'Industry not found'}), 404
    
    conn.commit()
    conn.close()
    
    return jsonify({'message': 'Industry updated successfully'})

@app.route('/api/industries/<industry_id>', methods=['DELETE'])
def delete_industry(industry_id):
    """Delete an industry"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if it's a default industry
    cursor.execute('SELECT is_default FROM industries WHERE id = ?', (industry_id,))
    row = cursor.fetchone()
    
    if not row:
        conn.close()
        return jsonify({'error': 'Industry not found'}), 404
    
    if row['is_default']:
        conn.close()
        return jsonify({'error': 'Cannot delete default industry'}), 400
    
    cursor.execute('DELETE FROM industries WHERE id = ?', (industry_id,))
    conn.commit()
    conn.close()
    
    return jsonify({'message': 'Industry deleted successfully'})

# OCR API - Synchronous for Vercel
@app.route('/api/ocr/batch', methods=['POST'])
def batch_ocr():
    """Submit multiple images and process synchronously (Vercel-friendly)"""
    data = request.get_json()
    
    if not data or not data.get('images'):
        return jsonify({'error': 'No images provided'}), 400
    
    images = data['images']
    results = []
    
    for image_data in images:
        try:
            # Convert base64 to proper format
            if ',' in image_data:
                base64_data = image_data.split(',')[1]
            else:
                base64_data = image_data
            
            # Process synchronously
            extracted_data = process_ocr_sync(base64_data)
            results.append(extracted_data)
                
        except Exception as e:
            results.append({'error': str(e)})
    
    # Merge results (combine data from multiple images)
    merged = {
        'fullName': '',
        'company': '',
        'designation': '',
        'phone': '',
        'email': '',
        'website': ''
    }
    
    for r in results:
        if isinstance(r, dict):
            for key in merged:
                if not merged[key] and r.get(key):
                    merged[key] = r[key]
    
    return jsonify(merged)

# File upload endpoint - base64 only (no filesystem writes in Vercel)
@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file uploads - returns base64 for storage in SQLite"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Read and encode as base64 directly (no filesystem writes)
    image_base64 = base64.b64encode(file.read()).decode('utf-8')
    
    # Run OCR synchronously
    result = process_ocr_sync(image_base64)
    
    return jsonify({
        'filename': file.filename,
        'image_base64': image_base64,
        'extracted': result
    })

# Export API
@app.route('/api/export', methods=['GET'])
def export_excel():
    """Export contacts to Excel format (CSV for simplicity)"""
    import csv
    import io
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM contacts ORDER BY created_at DESC')
    rows = cursor.fetchall()
    conn.close()
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow(['Name', 'Company', 'Designation', 'Phone', 'Email', 'Website', 'Industry', 'Created At'])
    
    # Data
    for row in rows:
        writer.writerow([
            row['full_name'],
            row['company'],
            row['designation'],
            row['phone'],
            row['email'],
            row['website'],
            row['industry'],
            row['created_at']
        ])
    
    output.seek(0)
    
    from flask import Response
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=BSM_Contacts.csv'}
    )

# Health check
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'database': os.path.exists(DATABASE_PATH),
        'vercel_mode': True
    })

# Initialize database on startup
init_database()

# Vercel handler (required for @vercel/python)
def handler(request):
    """Vercel serverless handler"""
    return app.full_dispatch

# Run locally
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

