# Database Migration Guide

## Overview
Sistem Kraepelin sekarang menggunakan PostgreSQL database untuk menyimpan:
- Metadata project (nama, tanggal lahir, pendidikan, dll)
- Preview history (hasil OCR digit recognition)
- Evaluation history (hasil evaluasi Kraepelin)

## Auto Migration
Database tables akan otomatis dibuat saat aplikasi pertama kali dijalankan. Tidak perlu menjalankan migration script secara manual.

## Database Schema

### Table: kraepelin_projects
Menyimpan metadata peserta tes
- id (Primary Key)
- filename (Unique, Indexed)
- name
- occupacy_and_role
- last_edu
- pob (Place of Birth)
- dob (Date of Birth)
- test_date
- created_at
- updated_at

### Table: preview_history
Menyimpan hasil OCR/digit recognition
- id (Primary Key)
- filename (Unique, Indexed)
- questions (JSON)
- answers (JSON)
- total_questions
- total_answers
- created_at
- updated_at

### Table: eval_history
Menyimpan hasil evaluasi Kraepelin
- id (Primary Key)
- filename (Unique, Indexed)
- panker
- tianker
- janker
- jankerv2
- hanker
- accuracy
- col_score_per_minute
- total_correct_ans
- plot_image_path
- created_at
- updated_at

### Table: draft_history
Menyimpan draft pekerjaan (sebelumnya di localStorage browser)
- id (Primary Key)
- filename (Unique, Indexed)
- draft_data (JSON)
- created_at
- updated_at

## Environment Variables
Tambahkan ke file `.env`:
```
DB_HOST=postgres
DB_PORT=5432
DB_NAME=kraepelin
DB_USER=kraepelin_user
DB_PASSWORD=kraepelin_pass123
```

## Docker Compose
PostgreSQL service sudah ditambahkan di `docker-compose.yml` dengan:
- Health check untuk memastikan database siap sebelum backend start
- Volume persistence untuk data
- Network configuration

## Fallback Mechanism
Semua controller memiliki fallback ke file system jika database gagal:
1. Coba baca dari database
2. Jika gagal atau tidak ada, baca dari file system
3. Data tetap disimpan ke kedua tempat (database + file system) untuk redundancy

## Testing Database Connection
```bash
# Masuk ke container postgres
docker exec -it kraepelin-postgres psql -U kraepelin_user -d kraepelin

# List tables
\dt

# Check data
SELECT * FROM kraepelin_projects;
SELECT * FROM preview_history;
SELECT * FROM eval_history;
```

## Troubleshooting

### Database connection error
- Pastikan PostgreSQL container sudah running: `docker ps`
- Check logs: `docker logs kraepelin-postgres`
- Verify environment variables di `.env`

### Tables not created
- Check backend logs: `docker logs kraepelin-backend`
- Database akan auto-create tables saat aplikasi start
- Jika masih error, restart backend: `docker restart kraepelin-backend`

### Data tidak tersimpan
- Check database connection
- Verify write permissions
- Check backend logs untuk error messages
