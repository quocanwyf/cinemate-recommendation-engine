import os
import pickle
import json
import pandas as pd
import scipy.sparse
import requests
from sqlalchemy import create_engine, text
from surprise import SVD, Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials

# --- 1. CẤU HÌNH BIẾN MÔI TRƯỜNG ---
DB_URL = os.getenv("DATABASE_URL")
RENDER_DEPLOY_HOOK = os.getenv("RENDER_DEPLOY_HOOK")
GDRIVE_JSON_STR = os.getenv("GDRIVE_SERVICE_ACCOUNT_JSON")

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
    print("🚀 BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN TỰ ĐỘNG...")
    engine = create_engine(DB_URL)
    drive = get_drive_instance()

    # --- 2. TẢI DỮ LIỆU (SỬA LỖI PRISMA Ở ĐÂY) ---
    print("📥 Đang lấy dữ liệu từ Database...")
    
    # Sửa lỗi 1: Dùng đúng tên cột userId, movieId, score và bọc tên bảng trong dấu ""
    query_ratings = 'SELECT "userId" as user_id, "movieId" as movie_id, "score" as rating FROM "Rating"'
    df_ratings = pd.read_sql(query_ratings, engine)

    # Sửa lỗi 2: Dùng JOIN để lấy Genres từ bảng Genre thông qua MovieGenre
    query_movies = """
        SELECT m.id, m.title, m.overview, 
               COALESCE(STRING_AGG(g.name, ' '), '') as genres
        FROM "Movie" m
        LEFT JOIN "MovieGenre" mg ON m.id = mg."movieId"
        LEFT JOIN "Genre" g ON g.id = mg."genreId"
        GROUP BY m.id, m.title, m.overview
    """
    df_movies = pd.read_sql(query_movies, engine)

    # --- 3. HUẤN LUYỆN MODEL ---
    print(f"🧠 Đang huấn luyện với {len(df_ratings)} ratings...")
    
    # Huấn luyện SVD
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(df_ratings[['user_id', 'movie_id', 'rating']], reader)
    model_svd = SVD(n_factors=100, n_epochs=20, random_state=42)
    model_svd.fit(data.build_full_trainset())

    # Huấn luyện TF-IDF
    # Kết hợp Overview và Genres thành một chuỗi văn bản để máy học
    df_movies['content'] = df_movies['overview'].fillna('') + ' ' + df_movies['genres']
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df_movies['content'])

    # --- 4. LƯU VÀ UPLOAD ---
    print("💾 Lưu file và đẩy lên Google Drive...")
    if not os.path.exists('models'): 
        os.makedirs('models')
    
    # Lưu file tạm thời vào thư mục models/ trên GitHub Runner
    with open('models/svd_model_v1.pkl', 'wb') as f: pickle.dump(model_svd, f)
    with open('models/tfidf_vectorizer.pkl', 'wb') as f: pickle.dump(tfidf, f)
    scipy.sparse.save_npz('models/tfidf_matrix.npz', tfidf_matrix)
    df_movies[['id', 'title']].to_pickle('models/movie_map.pkl')

    # Upload ghi đè lên các ID cũ trên Drive
    for name, f_id in FILE_IDS.items():
        print(f"   + Đang cập nhật: {name}")
        file_drive = drive.CreateFile({'id': f_id})
        file_drive.SetContentFile(os.path.join('models', name))
        file_drive.Upload()

    # --- 5. DEPLOY ---
    if RENDER_DEPLOY_HOOK:
        print("🔔 Gửi tín hiệu kích hoạt Deploy tới Render...")
        try:
            requests.post(RENDER_DEPLOY_HOOK)
            print("✅ Tín hiệu đã gửi thành công!")
        except Exception as e:
            print(f"❌ Lỗi gửi Webhook: {e}")
    
    print("🏁 HOÀN TẤT: Hệ thống đã được làm mới!")

if __name__ == "__main__":
    main()