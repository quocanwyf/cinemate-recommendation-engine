import os
import pickle
import json
import datetime
import pandas as pd
import scipy.sparse
import requests
import sys
from sqlalchemy import create_engine
from surprise import SVD, Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials

# --- 1. CẤU HÌNH ---
DB_URL = os.getenv("DATABASE_URL")
RENDER_DEPLOY_HOOK = os.getenv("RENDER_DEPLOY_HOOK")
GDRIVE_JSON_STR = os.getenv("GDRIVE_SERVICE_ACCOUNT_JSON")

THRESHOLD_RATINGS = 500 
DAYS_LIMIT = 7

FILE_IDS = {
    'svd_model_v1.pkl': '1hRs0q3X1lIGhuSedHI04V2lLXNa4bfal',
    'tfidf_matrix.npz': '1iAotmE9Qi6yTcAhoeGI03_BknCbE6BrL',
    'tfidf_vectorizer.pkl': '1YAWCkBomR0MQW9p2pxETSkZ1hfCN_Irf',
    'movie_map.pkl': '1raAjz3LVu5M6Z2yuO6QBUH3mSWU6IXgs',
    'metadata.json': '10S35CJuqVHj5L8NF_HjrXqcJzDbfDddg' 
}

def get_drive_instance():
    gauth = GoogleAuth()
    scope = ['https://www.googleapis.com/auth/drive']
    key_dict = json.loads(GDRIVE_JSON_STR)
    gauth.credentials = ServiceAccountCredentials.from_json_keyfile_dict(key_dict, scope)
    return GoogleDrive(gauth)

def check_should_retrain(engine, drive):
    print("🔍 Bước 1: Kiểm tra điều kiện Retrain...")
    current_count = pd.read_sql("SELECT count(*) FROM ratings", engine).iloc[0, 0]
    
    # Tải metadata từ Drive
    meta_file = drive.CreateFile({'id': FILE_IDS['metadata.json']})
    meta_file.GetContentFile('metadata.json')
    
    with open('metadata.json', 'r') as f:
        meta = json.load(f)
        last_count = meta.get('last_count', 0)
        last_date = datetime.datetime.strptime(meta.get('last_date', '2000-01-01'), '%Y-%m-%d').date()

    diff_ratings = current_count - last_count
    diff_days = (datetime.date.today() - last_date).days

    print(f"   + Rating mới: {diff_ratings}/{THRESHOLD_RATINGS}")
    print(f"   + Số ngày đã qua: {diff_days}/{DAYS_LIMIT}")

    if diff_ratings >= THRESHOLD_RATINGS or diff_days >= DAYS_LIMIT:
        print("✅ Thỏa mãn điều kiện! Bắt đầu huấn luyện...")
        return True, current_count
    return False, current_count

def main():
    engine = create_engine(DB_URL)
    drive = get_drive_instance()

    should_train, current_count = check_should_retrain(engine, drive)
    if not should_train:
        print("⏭️ Chưa đủ điều kiện. Kết thúc script.")
        return

    # --- 2. HUẤN LUYỆN (Kết hợp logic từ Notebook) ---
    print("🧠 Bước 2: Đang lấy dữ liệu và huấn luyện...")
    df_ratings = pd.read_sql("SELECT user_id, movie_id, rating FROM ratings", engine)
    df_movies = pd.read_sql("SELECT id, title, overview, genres FROM movies", engine)

    # Tiền xử lý SVD
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(df_ratings[['user_id', 'movie_id', 'rating']], reader)
    model_svd = SVD(n_factors=100, n_epochs=20, random_state=42).fit(data.build_full_trainset())

    # Tiền xử lý TF-IDF
    df_movies['content'] = df_movies['overview'].fillna('') + ' ' + df_movies['genres'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df_movies['content'])

    # --- 3. LƯU CỤC BỘ ---
    print("💾 Bước 3: Lưu file model cục bộ...")
    if not os.path.exists('models'): os.makedirs('models')
    
    with open('models/svd_model_v1.pkl', 'wb') as f: pickle.dump(model_svd, f)
    with open('models/tfidf_vectorizer.pkl', 'wb') as f: pickle.dump(tfidf, f)
    scipy.sparse.save_npz('models/tfidf_matrix.npz', tfidf_matrix)
    df_movies[['id', 'title']].to_pickle('models/movie_map.pkl')

    # --- 4. UPLOAD LÊN DRIVE (Gồm cả 4 model và metadata) ---
    print("☁️ Bước 4: Đẩy dữ liệu lên Google Drive...")
    
    # Upload 4 file models chính
    model_files = ['svd_model_v1.pkl', 'tfidf_matrix.npz', 'tfidf_vectorizer.pkl', 'movie_map.pkl']
    for name in model_files:
        print(f"   + Đang ghi đè: {name}")
        f_drive = drive.CreateFile({'id': FILE_IDS[name]})
        f_drive.SetContentFile(os.path.join('models', name))
        f_drive.Upload()

    # Cập nhật và Upload metadata.json (Bước chốt hạ)
    print("   + Đang cập nhật metadata.json...")
    new_meta = {'last_count': int(current_count), 'last_date': str(datetime.date.today())}
    with open('metadata.json', 'w') as f: json.dump(new_meta, f)
    
    meta_drive = drive.CreateFile({'id': FILE_IDS['metadata.json']})
    meta_drive.SetContentFile('metadata.json')
    meta_drive.Upload()

    # --- 5. DEPLOY ---
    if RENDER_DEPLOY_HOOK:
        print("🚀 Bước 5: Gửi tín hiệu Deploy tới Render...")
        requests.post(RENDER_DEPLOY_HOOK)
    
    print("🏁 HOÀN TẤT TOÀN BỘ QUY TRÌNH!")

if __name__ == "__main__":
    main()