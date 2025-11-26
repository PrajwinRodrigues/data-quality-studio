# backend/main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import math
import traceback
import os
import tempfile
from typing import Optional
from pydantic import BaseModel
import secrets
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from fastapi.security import OAuth2PasswordBearer
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends




app = FastAPI()



# ===================== AUTH / USERS SETUP =====================

DATABASE_URL = "sqlite:///./users.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)

Base.metadata.create_all(bind=engine)

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


SECRET_KEY = "change-me-in-production-very-secret"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    plain_password = plain_password[:72]
    return pwd_context.verify(plain_password, hashed_password)



def get_password_hash(password: str) -> str:
    # bcrypt max 72 bytes
    password = password[:72]
    return pwd_context.hash(password)




def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email).first()


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    name: str


class UserOut(BaseModel):
    id: int
    email: str
    name: str

    model_config = {
        "from_attributes": True
    }


class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserOut


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        sub = payload.get("sub")
        if sub is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.id == int(sub)).first()
    if user is None:
        raise credentials_exception
    return user



# allow Vite dev server to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.post("/auth/register", response_model=Token)
async def register_user(payload: UserCreate, db: Session = Depends(get_db)):
    existing = get_user_by_email(db, payload.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        email=payload.email,
        name=payload.name,
        hashed_password=get_password_hash(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": str(user.id)})
    return Token(
    access_token=token,
    token_type="bearer",
    user=UserOut.model_validate(user),
    )



@app.post("/auth/login", response_model=Token)
async def login_user(payload: LoginRequest, db: Session = Depends(get_db)):
    user = get_user_by_email(db, payload.email)
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({"sub": str(user.id)})
    return Token(
        access_token=token,
        token_type="bearer",
        user=user,
    )


@app.get("/auth/me", response_model=UserOut)
async def read_me(current_user: User = Depends(get_current_user)):
    return current_user



def save_upload_to_tempfile(upload: UploadFile) -> str:
    fd, path = tempfile.mkstemp(suffix=os.path.splitext(upload.filename)[1] or ".csv")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(upload.file.read())
    return path

def df_to_preview_rows(df: pd.DataFrame, n: int = 10):
    preview = df.head(n).fillna(value=np.nan).to_dict(orient="records")
    return preview

@app.post("/login")
async def login(req: LoginRequest):
    user = fake_users_db.get(req.username)
    if not user or user["password"] != req.password:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = secrets.token_hex(16)  # random hex string
    return {
        "access_token": token,
        "token_type": "bearer",
        "username": req.username,
    }

def df_to_summary(df: pd.DataFrame):
    summary = {}
    for col in df.columns:
        ser = df[col]
        dtype = str(ser.dtype)
        missing = int(ser.isna().sum())
        unique = int(ser.nunique(dropna=True))
        top_vals = ser.dropna().value_counts().head(5).to_dict()
        summary[col] = {
            "dtype": dtype,
            "missing_count": missing,
            "unique_count": unique,
            "top_values": top_vals,
        }
    return summary

def sanitize_for_json(obj):
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set, pd.Index)):
        return [sanitize_for_json(v) for v in list(obj)]

    if isinstance(obj, pd.Series):
        return [sanitize_for_json(v) for v in obj.tolist()]

    if isinstance(obj, (np.ndarray,)):
        return [sanitize_for_json(v) for v in obj.tolist()]

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, int) and not isinstance(obj, bool):
        return obj

    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        if math.isnan(val) or not math.isfinite(val):
            return None
        return val

    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)

    try:
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
    except Exception:
        pass

    if hasattr(obj, "item"):
        try:
            return sanitize_for_json(obj.item())
        except Exception:
            pass

    if isinstance(obj, str):
        return obj

    try:
        enc = jsonable_encoder(obj)
        return sanitize_for_json(enc)
    except Exception:
        return str(obj)


@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        saved_path = save_upload_to_tempfile(file)
        df = pd.read_csv(saved_path)
        before_preview = df_to_preview_rows(df, n=10)
        summary = df_to_summary(df)
        n_rows_total = int(len(df))
        columns = list(df.columns)

        result = {
            "columns": columns,
            "preview": before_preview,
            "summary": summary,
            "n_rows_total": n_rows_total,
            "saved_path": saved_path,
        }

        safe_result = sanitize_for_json(result)
        return JSONResponse(content=jsonable_encoder(safe_result))
    except Exception as e:
        tb = traceback.format_exc()
        print("=== UPLOAD CSV ERROR TRACEBACK ===")
        print(tb)
        return JSONResponse(
            status_code=500,
            content=jsonable_encoder({
                "detail": f"Upload error: {str(e)}",
                "traceback": tb[:5000]
            })
        )
    finally:
        try:
            file.file.close()
        except Exception:
            pass


@app.post("/suggest-rules")
async def suggest_rules(file: UploadFile = File(...)):
    try:
        saved_path = save_upload_to_tempfile(file)
        df = pd.read_csv(saved_path)

        suggestions = {}
        for col in df.columns:
            ser = df[col]
            n = len(ser)
            missing = int(ser.isna().sum())
            col_sugs = []
            coerced = pd.to_numeric(ser, errors="coerce")
            parsed_ok = coerced.notna().sum()
            if parsed_ok >= max(1, n // 2):
                col_sugs.append({"op": "parse_numeric", "reason": f"{parsed_ok}/{n} values parse as numbers"})
            if missing > 0:
                col_sugs.append({"op": "fill_missing", "reason": f"{missing}/{n} missing values"})
            if ser.nunique(dropna=True) <= max(1, n // 4):
                col_sugs.append({"op": "dedupe_by_cols", "reason": f"{ser.nunique(dropna=True)}/{n} unique (duplicates) "})
            if col_sugs:
                suggestions[col] = col_sugs

        safe = sanitize_for_json({"suggestions": suggestions})
        return JSONResponse(content=jsonable_encoder(safe))
    except Exception as e:
        tb = traceback.format_exc()
        print("=== SUGGEST RULES ERROR ===")
        print(tb)
        return JSONResponse(status_code=500, content=jsonable_encoder({"detail": str(e), "traceback": tb[:4000]}))


@app.post("/preview-rule")
async def preview_rule(file: Optional[UploadFile] = File(None), rule: str = Form(...)):
    try:
        if file is None:
            raise HTTPException(status_code=400, detail="file (UploadFile) is required")

        saved_path = save_upload_to_tempfile(file)
        df = pd.read_csv(saved_path)
        before_preview = df_to_preview_rows(df, n=10)

        import json
        try:
            rule_obj = json.loads(rule)
            op = rule_obj.get("op")
            col = rule_obj.get("col")
            cols = rule_obj.get("cols")
        except Exception:
            op = rule
            col = None
            cols = None

        df_after = df.copy()

        if op == "parse_numeric":
            if col:
                df_after[col] = pd.to_numeric(df_after[col], errors="coerce")
            else:
                for c in df_after.columns:
                    try:
                        df_after[c] = pd.to_numeric(df_after[c], errors="coerce")
                    except Exception:
                        pass
        elif op == "strip_spaces":
            if col:
                df_after[col] = df_after[col].astype(str).str.strip().replace("nan", None)
        elif op == "lowercase":
            if col:
                df_after[col] = df_after[col].astype(str).str.lower().replace("nan", None)
        elif op == "parse_date":
            if col:
                df_after[col] = pd.to_datetime(df_after[col], errors="coerce")
        elif op == "fill_missing":
            strategy = "auto"
            if isinstance(rule, str) and "strategy" in rule:
                try:
                    rule_json = json.loads(rule)
                    strategy = rule_json.get("strategy", "auto")
                except Exception:
                    pass
            if col:
                if pd.api.types.is_numeric_dtype(df_after[col]):
                    if strategy == "zero":
                        df_after[col] = df_after[col].fillna(0)
                    else:
                        df_after[col] = df_after[col].fillna(df_after[col].median() if df_after[col].notna().any() else 0)
                else:
                    df_after[col] = df_after[col].fillna("")
            else:
                for c in df_after.columns:
                    if pd.api.types.is_numeric_dtype(df_after[c]):
                        df_after[c] = df_after[c].fillna(0)
                    else:
                        df_after[c] = df_after[c].fillna("")
        elif op == "dedupe_by_cols":
            if cols:
                df_after = df_after.drop_duplicates(subset=cols)
            else:
                df_after = df_after.drop_duplicates()

        after_preview = df_to_preview_rows(df_after, n=10)
        result = {"before_preview": before_preview, "after_preview": after_preview}
        safe_result = sanitize_for_json(result)
        return JSONResponse(content=jsonable_encoder(safe_result))
    except Exception as e:
        tb = traceback.format_exc()
        print("=== PREVIEW RULE ERROR TRACEBACK ===")
        print(tb)
        return JSONResponse(status_code=500, content=jsonable_encoder({"detail": str(e), "traceback": tb[:4000]}))


@app.post("/replace-nans")
async def replace_nans(file: UploadFile = File(...), strategy: str = Form("auto")):
    try:
        saved_path = save_upload_to_tempfile(file)
        df = pd.read_csv(saved_path)
        df_out = df.copy()
        if strategy == "zero":
            for c in df_out.columns:
                if pd.api.types.is_numeric_dtype(df_out[c]):
                    df_out[c] = df_out[c].fillna(0)
                else:
                    df_out[c] = df_out[c].fillna("")
        else:  # auto
            for c in df_out.columns:
                if pd.api.types.is_numeric_dtype(df_out[c]):
                    df_out[c] = df_out[c].fillna(df_out[c].median() if df_out[c].notna().any() else 0)
                else:
                    df_out[c] = df_out[c].fillna("")
        result = {"preview": df_to_preview_rows(df_out, n=10)}
        return JSONResponse(content=jsonable_encoder(sanitize_for_json(result)))
    except Exception as e:
        tb = traceback.format_exc()
        print("=== REPLACE NaNs ERROR ===")
        print(tb)
        return JSONResponse(status_code=500, content=jsonable_encoder({"detail": str(e), "traceback": tb[:3000]}))


@app.get("/health")
async def health():
    return {"status": "ok"}
