import os
import pickle
import json
import pandas as pd
import scipy.sparse
import requests
from sqlalchemy import create_engine
from surprise import SVD, Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials

# --- 1. CẤU HÌNH BIẾN MÔI TRƯỜNG ---
DB_URL = os.getenv("DATABASE_URL")
RENDER_DEPLOY_HOOK = os.getenv("RENDER_DEPLOY_HOOK")
GDRIVE_JSON_STR = os.getenv("GDRIVE_SERVICE_ACCOUNT_JSON")

# Danh sách 4 file model trên Drive (Bỏ metadata.json)
FILE_IDS = {
    'svd_model_v1.pkl': '1hRs0q3X1lIGhuSedHI04V2lLXNa4bfal',
    'tfidf_matrix.npz': '1iAotmE9Qi6yTcAhoeGI03_BknCbE6BrL',
    'tfidf_vectorizer.pkl': '1YAWCkBomR0MQW9p2pxETSkZ1hfCN_Irf',
    'movie_map.pkl': '1raAjz3LVu5M6Z2yuO6QBUH3mSWU6IXgs'
}

def get_drive_instance():
    gauth = GoogleAuth()
    scope = ['https://www.googleapis.com/auth/drive']
    key_dict = json.loads(GDRIVE_JSON_STR)
    gauth.credentials = ServiceAccountCredentials.from_json_keyfile_dict(key_dict, scope)
    return GoogleDrive(gauth)

def main():
    print("🚀 BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN TỰ ĐỘNG HÀNG NGÀY...")
    engine = create_engine(DB_URL)
    drive = get_drive_instance()

    # --- 2. TẢI DỮ LIỆU ---
    print("📥 Đang lấy dữ liệu từ Database...")
    # LƯU Ý: Kiểm tra tên bảng là 'Rating' hay 'ratings' để tránh lỗi hôm trước nhé!
    df_ratings = pd.read_sql("SELECT user_id, movie_id, rating FROM ratings", engine)
    df_movies = pd.read_sql("SELECT id, title, overview, genres FROM movies", engine)

    # --- 3. HUẤN LUYỆN MODEL ---
    print(f"🧠 Đang huấn luyện với {len(df_ratings)} ratings...")
    
    # Huấn luyện SVD (Collaborative)
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(df_ratings[['user_id', 'movie_id', 'rating']], reader)
    model_svd = SVD(n_factors=100, n_epochs=20, random_state=42)
    model_svd.fit(data.build_full_trainset())

    # Huấn luyện TF-IDF (Content-Based)
    df_movies['content'] = df_movies['overview'].fillna('') + ' ' + df_movies['genres'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df_movies['content'])

    # --- 4. LƯU VÀ UPLOAD ---
    print("💾 Lưu file và đẩy lên Google Drive...")
    if not os.path.exists('models'): 
        os.makedirs('models')
    
    # Lưu cục bộ
    with open('models/svd_model_v1.pkl', 'wb') as f: pickle.dump(model_svd, f)
    with open('models/tfidf_vectorizer.pkl', 'wb') as f: pickle.dump(tfidf, f)
    scipy.sparse.save_npz('models/tfidf_matrix.npz', tfidf_matrix)
    df_movies[['id', 'title']].to_pickle('models/movie_map.pkl')

    # Upload ghi đè lên Drive
    for name, f_id in FILE_IDS.items():
        print(f"   + Đang cập nhật: {name}")
        file_drive = drive.CreateFile({'id': f_id})
        file_drive.SetContentFile(os.path.join('models', name))
        file_drive.Upload()

    # --- 5. DEPLOY ---
    if RENDER_DEPLOY_HOOK:
        print("🔔 Gửi tín hiệu kích hoạt Deploy tới Render...")
        requests.post(RENDER_DEPLOY_HOOK)
    
    print("🏁 HOÀN TẤT: Model đã được làm mới và server đang khởi động lại!")

if __name__ == "__main__":
    main()