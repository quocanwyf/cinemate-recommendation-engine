# 🚀 DEPLOYMENT CHECKLIST & PRODUCTION GUIDE

## 📋 PRE-DEPLOYMENT CHECKLIST

### ✅ Phase 1: Local Development (DONE)
- [x] Notebook training hoàn tất với RMSE ~0.85
- [x] Model files exported đầy đủ trong `models/`
- [x] FastAPI server chạy local thành công
- [x] Endpoints test thành công với `curl`
- [x] Input validation đã implement
- [x] Error handling đầy đủ

### ✅ Phase 2: Code Quality
- [x] Type hints đầy đủ (Pydantic models)
- [x] Docstrings cho tất cả endpoints
- [x] Comments cho logic phức tạp
- [x] Logging đã cấu hình
- [ ] Unit tests (Optional nhưng nên có)
- [ ] Integration tests

### ⚠️ Phase 3: Performance & Scalability
- [ ] Load testing với Apache Bench/Locust
- [ ] Redis caching đã tích hợp
- [ ] Database connection pooling (nếu cần)
- [ ] Rate limiting đã cấu hình

### ⚠️ Phase 4: Security
- [ ] API key authentication (nếu cần)
- [ ] CORS đã cấu hình đúng
- [ ] Input sanitization
- [ ] Environment variables security

### ⚠️ Phase 5: Monitoring
- [ ] Health check endpoint working
- [ ] Logging to file/service (Sentry, etc.)
- [ ] Metrics collection (Prometheus)
- [ ] Alerting rules setup

---

## 🐳 DOCKER DEPLOYMENT

### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files (IMPORTANT!)
COPY models/ models/

# Copy application code
COPY main.py .
COPY model_info.json .

# Expose port
EXPOSE 8000

# Run with Gunicorn + Uvicorn workers
CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

### Build & Run
```bash
# Build image
docker build -t cinemate-recommendation-api .

# Run container
docker run -d -p 8000:8000 --name cinemate-ai cinemate-recommendation-api

# Check logs
docker logs cinemate-ai

# Test
curl http://localhost:8000/
```

---

## 🌐 NGINX REVERSE PROXY (Recommended)

```nginx
upstream recommendation_api {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;  # Load balancing
    server 127.0.0.1:8002;
}

server {
    listen 80;
    server_name api.cinemate.com;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req zone=api_limit burst=20 nodelay;

    location /ai/ {
        proxy_pass http://recommendation_api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # Timeout settings
        proxy_connect_timeout 10s;
        proxy_send_timeout 10s;
        proxy_read_timeout 10s;
    }
}
```

---

## 🔥 GUNICORN PRODUCTION CONFIG

### gunicorn.conf.py
```python
import multiprocessing

# Worker settings
workers = multiprocessing.cpu_count() * 2 + 1  # 4 cores → 9 workers
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50

# Binding
bind = "0.0.0.0:8000"

# Logging
accesslog = "/var/log/cinemate/access.log"
errorlog = "/var/log/cinemate/error.log"
loglevel = "info"

# Process naming
proc_name = "cinemate-recommendation-api"

# Timeout
timeout = 30
graceful_timeout = 30
keepalive = 2
```

### Run
```bash
gunicorn -c gunicorn.conf.py main:app
```

---

## 📊 MONITORING SETUP

### 1. Prometheus + Grafana

**Install Prometheus exporter:**
```bash
pip install prometheus-fastapi-instrumentator
```

**Add to main.py:**
```python
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(...)

# Add metrics endpoint
Instrumentator().instrument(app).expose(app)
```

**Scrape config (prometheus.yml):**
```yaml
scrape_configs:
  - job_name: 'cinemate-ai'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /metrics
```

### 2. Sentry Error Tracking

```bash
pip install sentry-sdk[fastapi]
```

```python
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn="your-sentry-dsn",
    integrations=[FastApiIntegration()],
    traces_sample_rate=0.1,
)
```

---

## 🗄️ REDIS CACHING STRATEGY

### NestJS Integration Example

```typescript
// recommendation-cache.service.ts
@Injectable()
export class RecommendationCacheService {
  constructor(@InjectRedis() private readonly redis: Redis) {}

  async getOrFetchRecommendations(userId: string, limit: number = 10) {
    // 1. Try cache first
    const cacheKey = `rec:${userId}:${limit}`;
    const cached = await this.redis.get(cacheKey);
    
    if (cached) {
      console.log('✅ Cache HIT');
      return JSON.parse(cached);
    }

    // 2. Call Python API
    console.log('⚠️ Cache MISS - Calling AI API...');
    const result = await axios.get(
      `http://localhost:8000/recommend/svd/${userId}`,
      { params: { top_n: limit } }
    );

    // 3. Cache for 1 hour
    await this.redis.setex(cacheKey, 3600, JSON.stringify(result.data));

    return result.data;
  }

  // Invalidate cache khi user rating mới
  async invalidateUserCache(userId: string) {
    const keys = await this.redis.keys(`rec:${userId}:*`);
    if (keys.length > 0) {
      await this.redis.del(...keys);
      console.log(`🗑️ Invalidated ${keys.length} cache entries for ${userId}`);
    }
  }
}
```

---

## 🧪 LOAD TESTING

### Apache Bench
```bash
# Test health check (warm up)
ab -n 1000 -c 10 http://localhost:8000/

# Test SVD batch endpoint
ab -n 100 -c 5 -p test_payload.json -T application/json \
   http://localhost:8000/recommend/svd/batch
```

### Locust (Python)
```python
# locustfile.py
from locust import HttpUser, task, between

class RecommendationUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def test_health(self):
        self.client.get("/")
    
    @task(1)
    def test_recommendations(self):
        user_id = "fef77507-5d6c-4ed1-9843-a2e699906acb"
        self.client.get(f"/recommend/svd/{user_id}?top_n=10")
```

Run:
```bash
locust -f locustfile.py --host=http://localhost:8000
# Open http://localhost:8089
```

---

## 🔐 ENVIRONMENT VARIABLES

Create `.env` file:
```bash
# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Model Paths
MODEL_DIR=models
SVD_MODEL_PATH=models/svd_model_v1.pkl
TFIDF_MATRIX_PATH=models/tfidf_matrix.npz
MOVIE_MAP_PATH=models/movie_map.pkl

# Redis (if using)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Monitoring
SENTRY_DSN=your-sentry-dsn
LOG_LEVEL=INFO

# Security
ALLOWED_ORIGINS=http://localhost:3000,https://cinemate.com
API_KEY_ENABLED=false
```

Load in main.py:
```python
from dotenv import load_dotenv
load_dotenv()

API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', 8000))
```

---

## 📈 SCALING STRATEGY

### Vertical Scaling (1 server)
```
1 Server → 4-8 GB RAM, 2-4 CPU cores
└── Gunicorn: 9 workers (4 cores * 2 + 1)
    └── Each worker can handle ~100 req/s
    └── Total: ~900 req/s
```

### Horizontal Scaling (Multiple servers)
```
Load Balancer (Nginx)
├── AI Server 1 (8 workers)
├── AI Server 2 (8 workers)
└── AI Server 3 (8 workers)

Total capacity: ~2,400 req/s
```

---

## 🚨 ALERTING RULES

### Prometheus Alerts
```yaml
groups:
  - name: cinemate_ai_alerts
    rules:
      - alert: HighResponseTime
        expr: http_request_duration_seconds > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "API response time too high"

      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Error rate > 5%"

      - alert: ModelNotLoaded
        expr: up{job="cinemate-ai"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Recommendation API is DOWN"
```

---

## ✅ FINAL DEPLOYMENT STEPS

### 1. Pre-deployment
```bash
# Run all tests
pytest tests/

# Check code quality
flake8 main.py
mypy main.py

# Security scan
pip-audit
```

### 2. Deploy
```bash
# Pull latest code
git pull origin main

# Update dependencies
pip install -r requirements.txt

# Restart service
sudo systemctl restart cinemate-ai

# Check status
sudo systemctl status cinemate-ai
```

### 3. Post-deployment
```bash
# Smoke test
curl https://api.cinemate.com/ai/

# Check logs
tail -f /var/log/cinemate/error.log

# Monitor metrics
# Open Grafana dashboard
```

---

## 📞 TROUBLESHOOTING

### Issue 1: Model file not found
```bash
# Check file exists
ls -la models/

# Check permissions
chmod 644 models/*.pkl

# Verify path in code
grep "svd_model_v1.pkl" main.py
```

### Issue 2: Slow response times
```bash
# Check worker count
ps aux | grep gunicorn

# Check memory usage
free -h

# Add Redis caching (see above)
```

### Issue 3: High error rate
```bash
# Check logs
tail -100 /var/log/cinemate/error.log

# Check Sentry dashboard
# Check input validation logs
```

---

**🎉 HỆ THỐNG ĐÃ SẴN SÀNG CHO PRODUCTION!**

**Next steps:**
1. ✅ Setup Redis caching
2. ✅ Configure monitoring
3. ✅ Run load tests
4. 🚀 Deploy và theo dõi metrics

---

**📅 Version:** 1.0  
**📝 Last Updated:** 2025-12-18
