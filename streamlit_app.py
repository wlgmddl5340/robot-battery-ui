import base64
import json
import mimetypes
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="LG ROBO CARE | 로보킹 키우기",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      [data-testid="stHeader"],
      [data-testid="stToolbar"],
      [data-testid="stDecoration"],
      [data-testid="stSidebar"] {display:none!important;}
      #MainMenu, footer {visibility:hidden!important;}
      .stApp{
        background:
          radial-gradient(circle at 18% 10%,#fff9ec 0,transparent 30%),
          radial-gradient(circle at 88% 88%,#ead2a8 0,transparent 28%),
          #eee5d8;
      }
      .block-container{max-width:100%;padding:8px 4px 18px;}
      iframe{border:0!important;border-radius:28px;}
    
/* ===== Home visual adjustment: bigger station + mission moved right ===== */
#homePage .house{
  left:76px!important;
  top:112px!important;
  width:94px!important;
  height:80px!important;
  border-radius:38px 38px 9px 9px!important;
  box-shadow:0 8px 14px rgba(54,36,23,.24)!important;
}
#homePage .house:before{
  left:27px!important;
  bottom:0!important;
  width:40px!important;
  height:44px!important;
  border-radius:20px 20px 0 0!important;
}
#homePage .house:after{
  top:13px!important;
  left:27px!important;
  font-size:8.5px!important;
  letter-spacing:.2px!important;
}
#homePage .mission{
  left:auto!important;
  right:10px!important;
  bottom:14px!important;
  width:96px!important;
  z-index:18!important;
}

</style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 맞춤 배터리 준비 결과 기록 연결부
# GitHub에는 아래 구조로 기록을 올리면 됩니다.
# data/home_model_predictions.csv
# data/zone_model_predictions.csv
#
# 사진첩 / 분실물 이미지 연결부 (시연용)
# assets/photos/        → 4번째 탭 "사진첩"에 표시되는 반려동물 사진 (png/jpg/jpeg/gif/webp)
# assets/lost_items/    → 4번째 탭 "오늘의 발견"의 분실물 사진 (없으면 이모지로 표시)
# 각 폴더에 선택적으로 captions.json 을 두면 파일명별 제목/장소/시간/설명을 지정할 수 있습니다.
#   { "cat1.jpg": {"title": "낮잠 자는 콩이", "place": "거실 소파", "time": "오늘 오후 1:20", "note": "햇살 아래에서 낮잠 중"} }
# captions.json 이 없으면 파일명(확장자 제외)이 제목으로 사용됩니다.
# 사진은 한 장당 1MB 이하로 줄여두면 로딩이 빠릅니다.
# ============================================================


BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_DIR = BASE_DIR / "data"

# ============================================================
# 최종 ML 결과파일 연결부
# ============================================================
# 새 구조:
# data/ml_output.csv
#
# 머신러닝팀 최종 파일은 1행 = 1개 zone 청소 예측 결과입니다.
# 소형/중형/대형은 각각 4/6/8개 zone으로 구성됩니다.
#
# 집 전체 필요 SOC는 zone별 필요 SOC(zone_soc_used_pct)를 합산해서 계산합니다.
# 주의: zone_target_soc_pct를 합산하지 않습니다.
# zone_target_soc_pct는 zone 1개를 단독 청소할 때의 안전마진 포함 목표치입니다.
#
# fallback:
# 예전 파일명으로 업로드해도 동작하도록 ml_output(2).csv도 같이 탐색합니다.
# 예전 home_model_predictions.csv / zone_model_predictions.csv도 fallback으로 유지합니다.
# ============================================================

ML_OUTPUT_PATH = DATA_DIR / "ml_output.csv"
ML_OUTPUT_ALT_PATH = DATA_DIR / "ml_output(2).csv"
HOME_PRED_PATH = DATA_DIR / "home_model_predictions.csv"
ZONE_PRED_PATH = DATA_DIR / "zone_model_predictions.csv"

ASSET_DIR = BASE_DIR / "assets"
PHOTO_DIR = ASSET_DIR / "photos"
LOST_DIR = ASSET_DIR / "lost_items"

# 데모용 현재 배터리. 실제 제품에서는 로봇/앱에서 받은 현재 배터리로 교체하면 됩니다.
CURRENT_SOC = 80

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def _folder_signature(folder: Path):
    """폴더 안 파일이 바뀌면 캐시가 자동으로 갱신되도록 시그니처를 만듭니다."""
    if not folder.exists():
        return "missing"
    parts = []
    for p in sorted(folder.iterdir()):
        try:
            parts.append(f"{p.name}:{p.stat().st_mtime_ns}:{p.stat().st_size}")
        except Exception:
            parts.append(p.name)
    return "|".join(parts)


@st.cache_data
def load_image_folder(folder_str: str, signature: str):
    folder = Path(folder_str)
    items = []
    if not folder.exists() or not folder.is_dir():
        return items
    captions = {}
    cap_path = folder / "captions.json"
    if cap_path.exists():
        try:
            captions = json.loads(cap_path.read_text(encoding="utf-8")) or {}
        except Exception:
            captions = {}
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            raw = p.read_bytes()
        except Exception:
            continue
        mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
        data = base64.b64encode(raw).decode("ascii")
        meta = captions.get(p.name) or captions.get(p.stem) or {}
        if not isinstance(meta, dict):
            meta = {"title": str(meta)}
        items.append({
            "name": p.name,
            "src": f"data:{mime};base64,{data}",
            "title": str(meta.get("title") or p.stem),
            "place": str(meta.get("place") or ""),
            "time": str(meta.get("time") or ""),
            "note": str(meta.get("note") or ""),
        })
    return items


def _is_valid(value):
    return value is not None and not pd.isna(value)


def _safe_text(row, candidates, default=""):
    if row is None:
        return default
    for col in candidates:
        if col in row.index and _is_valid(row[col]):
            return str(row[col])
    return default


def _safe_float(row, candidates, default=0.0):
    if row is None:
        return float(default)
    for col in candidates:
        if col in row.index and _is_valid(row[col]):
            try:
                return float(row[col])
            except Exception:
                pass
    return float(default)


def _safe_int(row, candidates, default=0):
    try:
        return int(round(_safe_float(row, candidates, default)))
    except Exception:
        return int(default)


def _soc_target(required_soc):
    return int(round(max(15, min(float(required_soc) + 15, 90))))


def _infer_mop(row, prefix=""):
    """cleaning_type 또는 cleaning_type_code 기반으로 물걸레 여부를 추정합니다."""
    text_candidates = [
        f"{prefix}cleaning_type" if prefix else "cleaning_type",
        "cleaning_type_first",
        "cleaning_type",
    ]
    code_candidates = [
        f"{prefix}cleaning_type_code" if prefix else "cleaning_type_code",
        "cleaning_type_code_first",
        "cleaning_type_code",
    ]

    txt = _safe_text(row, text_candidates, "").lower()
    if any(k in txt for k in ["물", "걸레", "mop", "wet"]):
        return True
    if any(k in txt for k in ["건식", "dry"]):
        return False

    code = _safe_float(row, code_candidates, 0)
    return int(round(code)) == 1


@st.cache_data
def load_prediction_csv(path: str):
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "global_run_id" in df.columns:
        df["global_run_id"] = df["global_run_id"].astype(str)
    return df


def _first_existing_csv(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def _make_demo_runs():
    runs = []
    demo_specs = [
        (18, 4), (24, 4),
        (28, 6), (34, 6), (42, 6),
        (54, 8), (63, 8), (72, 8),
    ]

    for area, zone_count in demo_specs:
        for mop_enabled in [False, True]:
            base = area * 0.72 + (7 if mop_enabled else 0)
            labels_by_count = {
                4: ["거실", "주방", "침실", "현관"],
                6: ["침실", "주방", "거실", "카펫", "현관", "다용도"],
                8: ["침실1", "침실2", "주방", "거실", "현관", "카펫", "서재", "다용도"],
            }
            floors = ["마루/타일", "장판/PVC", "저파일 러그", "고파일 카펫", "마루/타일", "장판/PVC", "저파일 러그", "마루/타일"]
            dirts = ["낮음", "보통", "높음", "낮음", "보통", "높음", "낮음", "보통"]
            weights = {
                4: [0.30, 0.24, 0.28, 0.18],
                6: [0.17, 0.19, 0.25, 0.15, 0.12, 0.12],
                8: [0.12, 0.11, 0.14, 0.20, 0.10, 0.12, 0.11, 0.10],
            }[zone_count]

            zones = []
            for i, w in enumerate(weights, start=1):
                required = max(1.2, base * w)
                dirt = dirts[i - 1]
                zones.append({
                    "scope": "zone",
                    "zone": i,
                    "label": f"{i}구역",
                    "globalRunId": f"demo_{area}_{'mop' if mop_enabled else 'dry'}",
                    "areaPyung": area,
                    "zoneCount": zone_count,
                    "cleaningAreaM2": round(area * 3.3058 * 0.82 * w, 1),
                    "requiredSoc": round(required, 1),
                    "targetSoc": _soc_target(required),
                    "modelName": "Final ML",
                    "cleaningType": "물걸레" if mop_enabled else "건식",
                    "cleaningTypeCode": 1 if mop_enabled else 0,
                    "mopEnabled": mop_enabled,
                    "obstacleLevel": "보통",
                    "obstacleLevelCode": 2,
                    "floorType": floors[i - 1],
                    "dirtLevel": dirt,
                    "dirtCode": 3 if dirt == "높음" else (2 if dirt == "보통" else 1),
                    "suctionMode": "자동",
                    "suctionCode": 2,
                    "needsRecharge": "무충전 완주 가능",
                    "chargeCount": 0,
                })

            home_required = round(sum(z["requiredSoc"] for z in zones), 1)
            home = {
                "scope": "home",
                "label": "집 전체",
                "globalRunId": f"demo_{area}_{'mop' if mop_enabled else 'dry'}",
                "areaPyung": area,
                "zoneCount": zone_count,
                "cleaningAreaM2": round(sum(z["cleaningAreaM2"] for z in zones), 1),
                "requiredSoc": home_required,
                "targetSoc": _soc_target(home_required),
                "modelName": "Final ML",
                "cleaningType": "물걸레" if mop_enabled else "건식",
                "cleaningTypeCode": 1 if mop_enabled else 0,
                "mopEnabled": mop_enabled,
                "obstacleLevel": "보통",
                "obstacleLevelCode": 2,
                "floorType": "혼합",
                "dirtLevel": "평균",
                "dirtCode": round(sum(z["dirtCode"] for z in zones) / len(zones), 3),
                "dirtMaxCode": max(z["dirtCode"] for z in zones),
                "suctionMode": "자동",
                "suctionCode": 2,
                "suctionMaxCode": 3,
                "needsRecharge": "무충전 완주 가능",
                "chargeCount": 0,
            }
            runs.append({
                "globalRunId": home["globalRunId"],
                "areaPyung": area,
                "zoneCount": zone_count,
                "mopEnabled": mop_enabled,
                "cleaningType": home["cleaningType"],
                "home": home,
                "zones": zones,
            })
    return runs


def _build_zone_from_final_row(zrow, idx):
    zone_no = _safe_int(zrow, ["zone"], idx)
    area_pyung = _safe_int(zrow, ["area_pyung"], 0)
    zone_count = _safe_int(zrow, ["zone_count"], 0)
    required = _safe_float(zrow, ["zone_soc_used_pct"], 0)
    target = _safe_float(zrow, ["zone_target_soc_pct"], _soc_target(required))
    mop_enabled = _infer_mop(zrow)
    cleaning_type = _safe_text(zrow, ["cleaning_type"], "물걸레" if mop_enabled else "건식")
    obstacle_code = _safe_float(zrow, ["obstacle_level_code"], 0)
    dirt_code = _safe_float(zrow, ["dirt_level_code"], 0)
    if not dirt_code:
        dirt_txt = _safe_text(zrow, ["dirt_level"], "")
        if "높" in dirt_txt or "많" in dirt_txt:
            dirt_code = 3
        elif "보통" in dirt_txt or "중" in dirt_txt:
            dirt_code = 2
        elif "낮" in dirt_txt or "적" in dirt_txt:
            dirt_code = 1
    suction_code = _safe_float(zrow, ["suction_mode_code"], 0)
    if not suction_code:
        suction_txt = _safe_text(zrow, ["effective_suction_mode"], "")
        if "터보" in suction_txt:
            suction_code = 4
        elif "강" in suction_txt:
            suction_code = 3
        elif "중" in suction_txt:
            suction_code = 2
        elif "약" in suction_txt:
            suction_code = 1

    return {
        "scope": "zone",
        "zone": zone_no,
        "label": f"{zone_no}구역",
        "globalRunId": _safe_text(zrow, ["global_run_id"], ""),
        "areaPyung": area_pyung,
        "zoneCount": zone_count,
        "cleaningAreaM2": round(_safe_float(zrow, ["zone_area_m2"], 0), 1),
        "requiredSoc": round(float(required), 1),
        "targetSoc": int(round(max(15, min(float(target), 90)))),
        "modelName": "Final ML",
        "cleaningType": cleaning_type,
        "cleaningTypeCode": 1 if mop_enabled else 0,
        "mopEnabled": mop_enabled,
        "obstacleLevel": _safe_text(zrow, ["obstacle_level"], ""),
        "obstacleLevelCode": round(float(obstacle_code), 3),
        "floorType": _safe_text(zrow, ["floor_type"], ""),
        "dirtLevel": _safe_text(zrow, ["dirt_level"], ""),
        "dirtCode": round(float(dirt_code), 3),
        "suctionMode": _safe_text(zrow, ["effective_suction_mode"], ""),
        "suctionCode": round(float(suction_code), 3),
        "zoneTimeMin": round(_safe_float(zrow, ["zone_time_min"], 0), 2),
        "zoneProgressPct": round(_safe_float(zrow, ["zone_progress_pct"], 0), 2),
        "socBeforeZone": round(_safe_float(zrow, ["soc_before_zone_pct"], 0), 2),
        "socAfterZone": round(_safe_float(zrow, ["soc_after_zone_pct"], 0), 2),
        "needsRecharge": _safe_text(zrow, ["needs_recharge"], ""),
        "chargeCount": _safe_int(zrow, ["charge_count"], 0),
        "isChargeZone": _safe_int(zrow, ["is_charge_zone"], 0),
        "firstChargeZone": _safe_int(zrow, ["first_charge_zone"], -1),
        "secondChargeZone": _safe_int(zrow, ["second_charge_zone"], -1),
        "firstChargeTargetSoc": round(_safe_float(zrow, ["first_charge_target_soc_pct"], 0), 2),
        "secondChargeTargetSoc": round(_safe_float(zrow, ["second_charge_target_soc_pct"], 0), 2),
    }


def _build_run_from_final_group(gid, zdf):
    zdf = zdf.copy()
    if "zone" in zdf.columns:
        zdf = zdf.sort_values("zone")

    zones = []
    for idx, (_, zrow) in enumerate(zdf.iterrows(), start=1):
        zones.append(_build_zone_from_final_row(zrow, idx))

    if not zones:
        return None

    first = zdf.iloc[0]
    area_pyung = _safe_int(first, ["area_pyung"], zones[0].get("areaPyung", 0))
    zone_count_from_csv = _safe_int(first, ["zone_count"], len(zones))
    zone_count = zone_count_from_csv or len(zones)
    mop_enabled = _infer_mop(first)
    cleaning_type = _safe_text(first, ["cleaning_type"], "물걸레" if mop_enabled else "건식")

    # 핵심 변경점:
    # 집 전체 필요 SOC는 각 zone의 zone_soc_used_pct 합산값입니다.
    # run_total_soc_pct가 있더라도 UI에서는 이 합산값을 우선 사용합니다.
    home_required = round(sum(float(z.get("requiredSoc", 0)) for z in zones), 1)
    home_target = _soc_target(home_required)

    dirt_values = [float(z.get("dirtCode", 0)) for z in zones if float(z.get("dirtCode", 0)) > 0]
    suction_values = [float(z.get("suctionCode", 0)) for z in zones if float(z.get("suctionCode", 0)) > 0]
    obstacle_values = [float(z.get("obstacleLevelCode", 0)) for z in zones if float(z.get("obstacleLevelCode", 0)) > 0]

    home = {
        "scope": "home",
        "label": "집 전체",
        "globalRunId": str(gid),
        "areaPyung": area_pyung,
        "zoneCount": zone_count,
        "cleaningAreaM2": round(sum(float(z.get("cleaningAreaM2", 0)) for z in zones), 1),
        "requiredSoc": home_required,
        "targetSoc": home_target,
        "modelName": "Final ML",
        "cleaningType": cleaning_type,
        "cleaningTypeCode": 1 if mop_enabled else 0,
        "mopEnabled": mop_enabled,
        "obstacleLevel": _safe_text(first, ["obstacle_level"], ""),
        "obstacleLevelCode": round(sum(obstacle_values) / len(obstacle_values), 3) if obstacle_values else 0,
        "floorType": "혼합",
        "dirtLevel": "평균",
        "dirtCode": round(sum(dirt_values) / len(dirt_values), 3) if dirt_values else 0,
        "dirtMaxCode": round(max(dirt_values), 3) if dirt_values else 0,
        "suctionMode": "자동",
        "suctionCode": round(sum(suction_values) / len(suction_values), 3) if suction_values else 0,
        "suctionMaxCode": round(max(suction_values), 3) if suction_values else 0,
        "runTotalSocFromCsv": round(_safe_float(first, ["run_total_soc_pct"], home_required), 3),
        "runTotalTimeMin": round(_safe_float(first, ["run_total_time_min"], 0), 2),
        "runEndSoc": round(_safe_float(first, ["run_end_soc_pct"], 0), 2),
        "needsRecharge": _safe_text(first, ["needs_recharge"], ""),
        "chargeCount": _safe_int(first, ["charge_count"], 0),
        "firstChargeZone": _safe_int(first, ["first_charge_zone"], -1),
        "secondChargeZone": _safe_int(first, ["second_charge_zone"], -1),
    }

    return {
        "globalRunId": str(gid),
        "areaPyung": area_pyung,
        "zoneCount": zone_count,
        "mopEnabled": mop_enabled,
        "cleaningType": cleaning_type,
        "home": home,
        "zones": zones,
    }




def _limit_final_ml_runs(ml_df, max_runs=96, per_bucket=2):
    """CSV 원본은 그대로 두고, 브라우저(HTML/JS)로 넘기는 run만 제한합니다.

    중요:
    - global_run_id 단위로 선택하므로 한 run의 모든 zone은 함께 유지됩니다.
    - area_pyung까지 버킷에 포함해 각 평수 데이터가 샘플링 과정에서 사라지지 않게 합니다.
    - Streamlit/Pandas는 전체 CSV를 읽을 수 있지만, 전체 run을 JSON으로 HTML에 삽입하면
      components.html() 문서가 지나치게 커질 수 있으므로 이 단계에서만 줄입니다.
    """
    if ml_df is None or len(ml_df) == 0 or "global_run_id" not in ml_df.columns:
        return ml_df

    df = ml_df.copy()
    df["global_run_id"] = df["global_run_id"].astype(str)

    agg = {"global_run_id": "first"}
    for col, fn in [
        ("area_pyung", "first"),
        ("zone_count", "first"),
        ("cleaning_type_code", "first"),
        ("cleaning_type", "first"),
        ("charge_count", "max"),
        ("run_total_soc_pct", "first"),
    ]:
        if col in df.columns:
            agg[col] = fn

    run_meta = df.groupby("global_run_id", sort=False).agg(agg)

    def _bucket_row(row):
        try:
            area = int(round(float(row.get("area_pyung", 0) or 0)))
        except Exception:
            area = 0

        try:
            zone_count = int(round(float(row.get("zone_count", 0) or 0)))
        except Exception:
            zone_count = 0

        if zone_count not in [4, 6, 8]:
            zone_count = 4 if area <= 24 else (6 if area <= 49 else 8)

        mop = 0
        if "cleaning_type_code" in row.index and _is_valid(row.get("cleaning_type_code")):
            try:
                mop = int(round(float(row.get("cleaning_type_code"))))
            except Exception:
                mop = 0
        else:
            txt = str(row.get("cleaning_type", "")).lower()
            mop = 1 if any(k in txt for k in ["물", "걸레", "mop", "wet"]) else 0

        try:
            charge_flag = 1 if float(row.get("charge_count", 0) or 0) > 0 else 0
        except Exception:
            charge_flag = 0

        # 평수까지 포함해야 findRun(areaPyung, mopEnabled)가 필요한 평수를 찾을 수 있습니다.
        return f"{area}_{zone_count}_{mop}_{charge_flag}"

    run_meta["_bucket"] = run_meta.apply(_bucket_row, axis=1)

    selected_ids = []
    for _, part in run_meta.groupby("_bucket", sort=True):
        if "run_total_soc_pct" in part.columns:
            part = part.sort_values("run_total_soc_pct")
        else:
            part = part.sort_index()

        if len(part) <= per_bucket:
            chosen = part.index.tolist()
        elif per_bucket <= 1:
            chosen = [part.index[len(part) // 2]]
        else:
            positions = [round(i * (len(part) - 1) / (per_bucket - 1)) for i in range(per_bucket)]
            chosen = part.iloc[positions].index.tolist()

        selected_ids.extend(chosen)

    selected_ids = list(dict.fromkeys(map(str, selected_ids)))

    # 버킷이 매우 많아도 HTML로 넘기는 run 수는 상한을 둡니다.
    if len(selected_ids) > max_runs:
        selected_ids = selected_ids[:max_runs]

    return df[df["global_run_id"].isin(selected_ids)].copy()


def make_prediction_payload_from_final_ml(ml_df):
    runs = []
    if ml_df is not None and len(ml_df) > 0 and "global_run_id" in ml_df.columns:
        ml_df = ml_df.copy()
        ml_df["global_run_id"] = ml_df["global_run_id"].astype(str)
        for gid, zdf in ml_df.groupby("global_run_id", sort=False):
            run = _build_run_from_final_group(gid, zdf)
            if run is not None:
                runs.append(run)

    data_status = "final_ml_csv" if runs else "demo"
    if not runs:
        runs = _make_demo_runs()

    area_options = sorted({r["areaPyung"] for r in runs if r.get("areaPyung")})
    zone_count_options = sorted({r.get("zoneCount") for r in runs if r.get("zoneCount")})
    mop_values = sorted({bool(r["mopEnabled"]) for r in runs})
    default_run = runs[0]

    return {
        "currentSoc": int(CURRENT_SOC),
        "runs": runs,
        "areaOptions": area_options,
        "zoneCountOptions": zone_count_options,
        "defaultAreaPyung": default_run["areaPyung"],
        "defaultZoneCount": default_run.get("zoneCount", len(default_run.get("zones", []))),
        "defaultMopEnabled": bool(default_run["mopEnabled"]),
        "dataStatus": data_status,
        "homeSocRule": "sum_zone_soc_used_pct",
    }


# -----------------------------
# 예전 2파일 구조 fallback
# -----------------------------
def _build_home_scenario(home_row):
    required = _safe_float(
        home_row,
        ["best_pred_required_soc_pct", "pred_XGBoost", "pred_RandomForest", "home_required_soc_pct"],
        25,
    )
    target = _safe_float(
        home_row,
        ["best_pred_target_soc_pct", "home_target_soc_pct"],
        _soc_target(required),
    )
    mop_enabled = _infer_mop(home_row)
    cleaning_type = _safe_text(home_row, ["cleaning_type_first", "cleaning_type"], "물걸레" if mop_enabled else "건식")
    cleaning_type_code = int(round(_safe_float(home_row, ["cleaning_type_code_first", "cleaning_type_code"], 1 if mop_enabled else 0)))
    obstacle_code = _safe_float(home_row, ["obstacle_level_code_first", "obstacle_level_code"], 0)
    dirt_mean_code = _safe_float(home_row, ["dirt_level_code_mean", "dirt_level_code"], 0)
    dirt_max_code = _safe_float(home_row, ["dirt_level_code_max", "dirt_level_code"], dirt_mean_code)
    suction_mean_code = _safe_float(home_row, ["suction_mode_code_mean", "suction_mode_code"], 0)
    suction_max_code = _safe_float(home_row, ["suction_mode_code_max", "suction_mode_code"], suction_mean_code)
    zone_count = _safe_int(home_row, ["zone_count_first", "zone_count"], 0)
    return {
        "scope": "home",
        "label": "집 전체",
        "globalRunId": _safe_text(home_row, ["global_run_id"], ""),
        "areaPyung": int(round(_safe_float(home_row, ["area_pyung_first", "area_pyung"], 0))),
        "zoneCount": zone_count,
        "cleaningAreaM2": int(round(_safe_float(home_row, ["zone_area_m2_sum", "cleaning_area_m2"], 0))),
        "requiredSoc": round(float(required), 1),
        "targetSoc": int(round(max(15, min(float(target), 90)))),
        "modelName": _safe_text(home_row, ["best_model"], "XGBoost"),
        "cleaningType": cleaning_type,
        "cleaningTypeCode": cleaning_type_code,
        "mopEnabled": mop_enabled,
        "obstacleLevel": _safe_text(home_row, ["obstacle_level_first", "obstacle_level"], ""),
        "obstacleLevelCode": round(float(obstacle_code), 3),
        "floorType": "혼합",
        "dirtLevel": "평균",
        "dirtCode": round(float(dirt_mean_code), 3),
        "dirtMaxCode": round(float(dirt_max_code), 3),
        "suctionMode": "자동",
        "suctionCode": round(float(suction_mean_code), 3),
        "suctionMaxCode": round(float(suction_max_code), 3),
    }


def _build_zone_scenario(zrow, idx, home):
    zone_no = int(round(_safe_float(zrow, ["zone"], idx)))
    required = _safe_float(
        zrow,
        ["best_pred_required_soc_pct", "pred_RandomForest", "pred_XGBoost", "zone_required_soc_pct"],
        max(home["requiredSoc"] / 5, 1),
    )
    mop_enabled = _infer_mop(zrow)
    cleaning_type = _safe_text(zrow, ["cleaning_type"], home.get("cleaningType", ""))
    cleaning_type_code = int(round(_safe_float(zrow, ["cleaning_type_code"], 1 if mop_enabled else 0)))
    obstacle_code = _safe_float(zrow, ["obstacle_level_code"], home.get("obstacleLevelCode", 0))
    dirt_code = _safe_float(zrow, ["dirt_level_code"], 0)
    suction_code = _safe_float(zrow, ["suction_mode_code"], 0)
    return {
        "scope": "zone",
        "zone": zone_no,
        "label": f"{zone_no}구역",
        "globalRunId": _safe_text(zrow, ["global_run_id"], home["globalRunId"]),
        "areaPyung": int(round(_safe_float(zrow, ["area_pyung", "area_pyung_first"], home["areaPyung"]))),
        "zoneCount": int(home.get("zoneCount") or 0),
        "cleaningAreaM2": round(_safe_float(zrow, ["zone_area_m2"], 0), 1),
        "requiredSoc": round(float(required), 1),
        "targetSoc": _soc_target(required),
        "modelName": _safe_text(zrow, ["best_model"], "RandomForest"),
        "cleaningType": cleaning_type,
        "cleaningTypeCode": cleaning_type_code,
        "mopEnabled": mop_enabled,
        "obstacleLevel": _safe_text(zrow, ["obstacle_level"], home.get("obstacleLevel", "")),
        "obstacleLevelCode": round(float(obstacle_code), 3),
        "floorType": _safe_text(zrow, ["floor_type"], ""),
        "dirtLevel": _safe_text(zrow, ["dirt_level"], ""),
        "dirtCode": round(float(dirt_code), 3),
        "suctionMode": _safe_text(zrow, ["effective_suction_mode"], ""),
        "suctionCode": round(float(suction_code), 3),
    }


def make_prediction_payload(home_df, zone_df):
    runs = []

    if home_df is not None and len(home_df) > 0:
        for _, hrow in home_df.iterrows():
            home = _build_home_scenario(hrow)
            gid = home["globalRunId"]
            zones = []

            if zone_df is not None and "global_run_id" in zone_df.columns and gid:
                zdf = zone_df[zone_df["global_run_id"].astype(str) == str(gid)].copy()
                if "zone" in zdf.columns:
                    zdf = zdf.sort_values("zone")
                for idx, (_, zrow) in enumerate(zdf.iterrows(), start=1):
                    zone = _build_zone_scenario(zrow, idx, home)
                    zone["cleaningType"] = home["cleaningType"]
                    zone["mopEnabled"] = home["mopEnabled"]
                    zones.append(zone)

            if len(zones) >= 1:
                # fallback에서도 zone_count가 없으면 실제 zone 개수를 사용
                home["zoneCount"] = int(home.get("zoneCount") or len(zones))
                for z in zones:
                    z["zoneCount"] = home["zoneCount"]
                runs.append({
                    "globalRunId": gid,
                    "areaPyung": home["areaPyung"],
                    "zoneCount": home["zoneCount"],
                    "mopEnabled": home["mopEnabled"],
                    "cleaningType": home["cleaningType"],
                    "home": home,
                    "zones": zones,
                })

    if not runs and zone_df is not None and len(zone_df) > 0 and "global_run_id" in zone_df.columns:
        for gid, zdf in zone_df.groupby("global_run_id"):
            if len(zdf) < 1:
                continue
            if "zone" in zdf.columns:
                zdf = zdf.sort_values("zone")
            first = zdf.iloc[0]
            area = int(round(_safe_float(first, ["area_pyung"], 18)))
            mop_enabled = _infer_mop(first)
            dummy_home = {
                "scope": "home", "label": "집 전체", "globalRunId": str(gid), "areaPyung": area,
                "zoneCount": len(zdf),
                "cleaningAreaM2": int(round(zdf["zone_area_m2"].sum())) if "zone_area_m2" in zdf.columns else 0,
                "requiredSoc": 0, "targetSoc": 15, "modelName": "XGBoost",
                "cleaningType": "물걸레" if mop_enabled else "건식", "cleaningTypeCode": 1 if mop_enabled else 0, "mopEnabled": mop_enabled,
                "obstacleLevel": _safe_text(first, ["obstacle_level"], ""), "obstacleLevelCode": _safe_float(first, ["obstacle_level_code"], 0), "floorType": "혼합",
                "dirtLevel": "평균", "dirtCode": 0, "dirtMaxCode": 0, "suctionMode": "자동", "suctionCode": 0, "suctionMaxCode": 0
            }
            zones = []
            dummy_required = 0
            for idx, (_, zrow) in enumerate(zdf.iterrows(), start=1):
                zone = _build_zone_scenario(zrow, idx, dummy_home)
                dummy_required += zone["requiredSoc"]
                zones.append(zone)
            dummy_home["requiredSoc"] = round(dummy_required, 1)
            dummy_home["targetSoc"] = _soc_target(dummy_required)
            runs.append({
                "globalRunId": str(gid), "areaPyung": area, "zoneCount": len(zones), "mopEnabled": mop_enabled,
                "cleaningType": dummy_home["cleaningType"], "home": dummy_home, "zones": zones
            })

    data_status = "legacy_csv" if runs else "demo"
    if not runs:
        runs = _make_demo_runs()

    area_options = sorted({r["areaPyung"] for r in runs if r.get("areaPyung")})
    zone_count_options = sorted({r.get("zoneCount") for r in runs if r.get("zoneCount")})
    mop_values = sorted({bool(r["mopEnabled"]) for r in runs})
    default_run = runs[0]

    return {
        "currentSoc": int(CURRENT_SOC),
        "runs": runs,
        "areaOptions": area_options,
        "zoneCountOptions": zone_count_options,
        "defaultAreaPyung": default_run["areaPyung"],
        "defaultZoneCount": default_run.get("zoneCount", len(default_run.get("zones", []))),
        "defaultMopEnabled": bool(default_run["mopEnabled"]),
        "dataStatus": data_status,
    }


ml_output_path = _first_existing_csv([ML_OUTPUT_PATH, ML_OUTPUT_ALT_PATH])
ml_output_df = load_prediction_csv(str(ml_output_path)) if ml_output_path else None

if ml_output_df is not None:
    # CSV 전체는 Pandas가 읽되, components.html() 안으로는 대표 run만 전달합니다.
    # 3~5만 행 자체보다 전체 데이터를 JSON으로 브라우저에 삽입하는 것이 훨씬 큰 병목입니다.
    ml_output_ui_df = _limit_final_ml_runs(ml_output_df, max_runs=96, per_bucket=2)
    ui_prediction_data = make_prediction_payload_from_final_ml(ml_output_ui_df)
else:
    home_pred_df = load_prediction_csv(str(HOME_PRED_PATH))
    zone_pred_df = load_prediction_csv(str(ZONE_PRED_PATH))
    ui_prediction_data = make_prediction_payload(home_pred_df, zone_pred_df)

UI_PREDICTION_JSON = json.dumps(ui_prediction_data, ensure_ascii=False)

ui_media_data = {
    "photos": load_image_folder(str(PHOTO_DIR), _folder_signature(PHOTO_DIR)),
    "lostItems": load_image_folder(str(LOST_DIR), _folder_signature(LOST_DIR)),
}
UI_MEDIA_JSON = json.dumps(ui_media_data, ensure_ascii=False)


APP_HTML = r"""
<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
:root{
  --brown:#4b3324;--brown2:#735139;--cream:#fff8e8;--green:#62aa49;
  --green2:#2f8b3a;--orange:#ff9535;--red:#ef4e45;--yellow:#ffd44f;
  --shadow:0 8px 18px rgba(72,44,20,.15)
}
html,body{margin:0;min-height:100%;font-family:"Apple SD Gothic Neo","Malgun Gothic","Noto Sans KR",sans-serif;color:var(--brown);background:transparent}
body{display:flex;justify-content:center;align-items:flex-start;padding:8px}
button,input{font-family:inherit} button{cursor:pointer}
.phone{position:relative;width:min(100%,420px);height:960px;overflow:hidden;border:8px solid #242321;border-radius:40px;background:#dfb46b;box-shadow:0 30px 80px rgba(50,33,18,.28),0 8px 20px rgba(50,33,18,.16)}
.notch{position:absolute;z-index:100;top:0;left:50%;width:126px;height:25px;transform:translateX(-50%);border-radius:0 0 18px 18px;background:#242321}
.screen{position:relative;width:100%;height:100%;overflow:hidden;background:linear-gradient(180deg,#d2ab7b 0%,#e8c793 44%,#e1b36c 100%)}

/* Header */
.header{position:relative;z-index:50;height:128px;padding:23px 14px 8px;color:#fff;background:linear-gradient(120deg,#2d2722,#5b3f2d);box-shadow:0 7px 16px rgba(48,31,19,.22)}
.header-top{display:flex;justify-content:space-between;align-items:center}
.brand{font-size:9px;font-weight:900;letter-spacing:1.4px}
.app-title{margin-top:3px;font-size:23px;font-weight:900}
.coin-pill{display:flex;align-items:center;gap:5px;padding:8px 12px;border:1px solid rgba(255,255,255,.17);border-radius:18px;background:rgba(255,255,255,.13);font-size:12px;font-weight:900}
.nav{display:grid;grid-template-columns:repeat(5,1fr);gap:6px;margin-top:12px}
.nav-btn{padding:8px 3px;border:0;border-radius:14px;background:transparent;color:rgba(255,255,255,.72);font-size:10px;font-weight:900;transition:.2s}
.nav-btn.active{background:#4d291c;color:#fff;box-shadow:0 4px 9px rgba(0,0,0,.18)}
.nav-btn:active{transform:scale(.95)}

/* Pages */
.pages{position:relative;height:calc(100% - 128px);overflow:hidden}
.page{position:absolute;inset:0;display:none;overflow-y:auto;padding:12px 10px 24px;animation:pageIn .25s ease-out}
.page.active{display:block}.page::-webkit-scrollbar{width:0}
@keyframes pageIn{from{opacity:0;transform:translateX(16px)}to{opacity:1;transform:none}}
.section-kicker{color:#76533b;font-size:9px;font-weight:900;letter-spacing:1.3px}
.section-title{margin:2px 0 11px;font-size:22px;font-weight:900}
.panel{border:1px solid rgba(136,87,40,.14);border-radius:17px;background:rgba(255,248,231,.96);box-shadow:var(--shadow)}

/* Home */
#homePage{padding:0;background:linear-gradient(180deg,#cfaa7d 0%,#e0c39a 39%,#d29c58 40%,#d09850 100%)}
.room{position:relative;height:355px;overflow:hidden}
.wall-light{position:absolute;top:-80px;left:50%;width:420px;height:280px;transform:translateX(-50%);border-radius:50%;background:radial-gradient(circle,rgba(255,248,226,.88),rgba(255,240,205,.18) 60%,transparent 74%)}
.floor{position:absolute;left:0;right:0;bottom:0;height:170px;background:repeating-linear-gradient(90deg,rgba(106,63,26,.1) 0,rgba(106,63,26,.1) 2px,transparent 2px,transparent 66px),repeating-linear-gradient(0deg,transparent 0,transparent 38px,rgba(106,63,26,.09) 39px,rgba(106,63,26,.09) 41px),linear-gradient(180deg,#d9b174,#c78e48)}
.plant{position:absolute;z-index:3;left:14px;top:70px;font-size:74px;filter:drop-shadow(0 7px 5px rgba(68,41,21,.18))}
.house{position:absolute;z-index:3;left:88px;top:128px;width:68px;height:58px;border-radius:29px 29px 7px 7px;background:linear-gradient(145deg,#a38b71,#74614d);box-shadow:0 6px 11px rgba(54,36,23,.22)}
.house:before{content:"";position:absolute;left:19px;bottom:0;width:31px;height:32px;border-radius:16px 16px 0 0;background:#403b35}
.house:after{content:"HOME";position:absolute;top:11px;left:16px;color:rgba(255,255,255,.56);font-size:7px;font-weight:900}
.sofa{position:absolute;z-index:2;right:-8px;top:72px;width:139px;height:105px;border-radius:28px 0 8px 8px;background:linear-gradient(145deg,#71847f,#354e4a);box-shadow:0 10px 14px rgba(54,37,23,.27)}
.sofa:before{content:"";position:absolute;top:-18px;left:10px;width:80px;height:48px;border-radius:17px 17px 7px 7px;background:#74867f}
.speech{position:absolute;z-index:20;top:17px;left:50%;width:177px;min-height:74px;padding:14px 12px;transform:translateX(-50%);border-radius:14px;background:rgba(255,255,255,.98);box-shadow:0 7px 16px rgba(58,37,21,.18);text-align:center;font-size:12px;line-height:1.55;font-weight:800}
.speech strong{color:var(--green);font-size:14px}
.speech:after{content:"";position:absolute;left:52%;bottom:-15px;border-width:16px 10px 0 4px;border-style:solid;border-color:white transparent transparent transparent}
.mode-chip{position:absolute;z-index:18;top:106px;left:50%;padding:7px 13px;transform:translateX(-50%);border-radius:18px;background:rgba(255,248,229,.95);box-shadow:0 4px 10px rgba(72,46,23,.16);font-size:9px;font-weight:900}
.rug{position:absolute;z-index:4;left:50%;bottom:12px;width:255px;height:92px;transform:translateX(-50%);border-radius:50%;background:radial-gradient(ellipse,rgba(241,224,197,.91),rgba(191,151,103,.79));box-shadow:inset 0 0 18px rgba(103,70,43,.14)}
.clean-path{position:absolute;z-index:5;left:50%;bottom:41px;width:292px;height:78px;transform:translateX(-50%);overflow:hidden;border:2px dashed rgba(255,255,255,.36);border-radius:50%;opacity:0}
.clean-fill{width:0;height:100%;border-radius:inherit;background:linear-gradient(90deg,rgba(95,177,105,.08),rgba(102,200,115,.51));transition:width .3s}
.charge-ring{position:absolute;z-index:7;left:50%;bottom:19px;width:216px;height:132px;transform:translateX(-50%);border:5px solid transparent;border-top-color:#ffd345;border-right-color:#ff9931;border-radius:50%;opacity:0}
.dust{position:absolute;z-index:8;left:50%;bottom:55px;width:267px;height:75px;transform:translateX(-50%);pointer-events:none}
.dust span{position:absolute;bottom:0;width:7px;height:7px;border-radius:50%;background:rgba(116,78,42,.5);opacity:0}
.dust span:nth-child(1){left:8%}.dust span:nth-child(2){left:21%;animation-delay:.32s}.dust span:nth-child(3){left:37%;animation-delay:.68s}.dust span:nth-child(4){right:8%;animation-delay:.18s}.dust span:nth-child(5){right:23%;animation-delay:.52s}.dust span:nth-child(6){right:39%;animation-delay:.88s}
.robot{position:absolute;z-index:10;left:50%;bottom:31px;width:181px;height:99px;transform:translateX(-50%);transform-origin:center bottom;border:2px solid #a29b92;border-radius:62% 62% 40% 40%;background:linear-gradient(180deg,#fffefb 0%,#e7e8e3 70%,#bbbdb7 100%);box-shadow:0 13px 19px rgba(55,37,21,.32),inset 0 -8px 12px rgba(83,83,79,.12);cursor:pointer;animation:robotIdle 2.8s ease-in-out infinite}
.robot-top{position:absolute;left:50%;top:-5px;width:126px;height:58px;transform:translateX(-50%);border-top:2px solid rgba(119,119,114,.42);border-radius:50%;background:radial-gradient(ellipse at center,#fbfbf8 0%,#d5d6d2 74%,#b8b9b3 100%)}
.face{position:absolute;z-index:3;left:50%;bottom:8px;width:128px;height:54px;transform:translateX(-50%);border-radius:26px 26px 30px 30px;background:linear-gradient(180deg,#2a2c2d,#101213 76%);box-shadow:inset 0 4px 5px rgba(255,255,255,.13)}
.eye{position:absolute;top:14px;width:22px;height:22px;border:2px solid #f2f0dc;border-radius:50%;background:#111;transform-origin:center;animation:blink 4.8s infinite}
.eye:after{content:"";position:absolute;top:4px;left:5px;width:7px;height:7px;border-radius:50%;background:#fff;transition:transform .25s}
.eye.left{left:20px}.eye.right{right:20px}.robot.look-left .eye:after{transform:translateX(-4px)}.robot.look-right .eye:after{transform:translateX(4px)}
.cheek{position:absolute;bottom:8px;width:13px;height:6px;border-radius:50%;background:#ff8d8d;opacity:.75}.cheek.left{left:8px}.cheek.right{right:8px}
.mouth{position:absolute;left:50%;bottom:8px;width:20px;height:11px;transform:translateX(-50%);border:2px solid #f3d6c9;border-top:0;border-radius:0 0 12px 12px}
.crown{position:absolute;z-index:12;left:50%;top:-40px;transform:translateX(-50%);font-size:46px;filter:drop-shadow(0 4px 3px rgba(81,52,17,.25))}
.spark{position:absolute;z-index:11;right:-16px;top:-9px;font-size:28px;animation:sparkle 1.3s ease-in-out infinite}
.slot{position:absolute;left:50%;bottom:-2px;width:43px;height:5px;transform:translateX(-50%);border-radius:10px;background:#484a48}
.effect-layer{position:absolute;z-index:25;inset:0;overflow:hidden;pointer-events:none}.effect{position:absolute;left:50%;top:68%;font-size:20px;animation:effectFly 1.2s ease-out forwards}
.room.cleaning .mode-chip{color:#fff;background:rgba(57,143,82,.93)}.room.cleaning .clean-path{opacity:1}.room.cleaning .dust span{animation:dustRise 1.35s ease-out infinite}.room.cleaning .robot{animation:robotPatrol 2.4s ease-in-out infinite}
.room.charging .mode-chip{color:#fff;background:rgba(242,145,35,.94)}.room.charging .charge-ring{opacity:.92;animation:ringSpin 1.05s linear infinite}.room.charging .robot{animation:robotCharge .85s ease-in-out infinite}
.room.low .robot{animation:robotLow .48s linear infinite}.room.celebrate .robot{animation:robotCelebrate .72s ease-in-out 3}.robot.tap{animation:robotTap .62s ease-out!important}
.mission{position:absolute;z-index:16;left:8px;bottom:14px;width:90px;padding:9px 7px;border-radius:14px;background:rgba(255,248,230,.97);box-shadow:0 4px 10px rgba(61,39,20,.22)}
.mission-title{font-size:10px;font-weight:900}.mission-text{margin-top:5px;font-size:9px;line-height:1.4;font-weight:800}.mission-progress{display:flex;align-items:center;gap:4px;margin-top:8px}.mission-track{flex:1;height:7px;overflow:hidden;border-radius:10px;background:#c9b794}.mission-fill{width:0;height:100%;border-radius:inherit;background:linear-gradient(90deg,#ff8e33,#ffca43);transition:width .3s}.reward-small{color:#8d5814;font-size:9px;font-weight:900}
.quick{position:absolute;z-index:17;right:7px;top:89px;display:flex;flex-direction:column;gap:8px}.quick-btn{width:52px;min-height:51px;padding:4px 2px;border:0;border-radius:15px;background:rgba(255,248,231,.97);box-shadow:0 4px 10px rgba(61,39,20,.22);color:var(--brown);font-size:8px;font-weight:900}.quick-btn .icon{display:block;margin-bottom:3px;font-size:21px}
.home-dashboard{padding:9px 8px 15px;background:linear-gradient(180deg,rgba(239,205,151,.99),rgba(227,181,110,.99))}
.actions{display:grid;grid-template-columns:repeat(6,1fr);gap:4px;margin-bottom:8px;padding:8px 5px;border:1px solid rgba(138,91,40,.18);border-radius:14px;background:rgba(255,247,224,.94);box-shadow:0 4px 9px rgba(87,54,25,.15)}
.action-btn{min-width:0;padding:3px 0;border:0;background:transparent;color:var(--brown);font-size:8px;font-weight:900}.action-icon{display:block;margin-bottom:4px;font-size:23px}
.home-cards{display:grid;grid-template-columns:1.16fr 1fr .78fr;gap:6px}.mini-card{min-height:186px;padding:11px 9px;border:1px solid rgba(139,92,39,.17);border-radius:14px;background:rgba(255,248,230,.97);box-shadow:0 4px 10px rgba(79,48,21,.12)}
.mini-title{margin-bottom:8px;font-size:10px;font-weight:900}.battery-info{font-size:9px;line-height:1.55;font-weight:800}.battery-face{margin:7px 0 5px;text-align:center;font-size:31px}
.scale{position:relative;height:7px;margin:18px 7px 14px;border-radius:10px;background:linear-gradient(90deg,#f15b44 0%,#ffd25c 44%,#76ad51 100%)}.pointer{position:absolute;top:-7px;left:81%;width:14px;height:14px;transform:translateX(-50%);border:3px solid #fff;border-radius:50%;background:#e64c39;box-shadow:0 2px 5px rgba(0,0,0,.23);transition:left .3s}
.scale-labels{display:flex;justify-content:space-between;font-size:7px;font-weight:900}.battery-message{margin-top:10px;padding:7px;border-radius:9px;background:#f1e4c6;font-size:8px;line-height:1.45;font-weight:800}
.time-card{text-align:center}.time-icon{margin-top:10px;font-size:31px;animation:float 2s ease-in-out infinite}.time-number{color:#ef573f;font-size:32px;line-height:1;font-weight:900}.time-number small{font-size:12px}.time-sub{margin-top:4px;font-size:8px;font-weight:800}.time-tip{margin-top:15px;padding:7px 5px;border-radius:9px;background:#fff0ca;color:#745431;font-size:8px;line-height:1.45;font-weight:800}
.food-card{display:flex;flex-direction:column;align-items:center;justify-content:space-between}.food-title{width:100%;font-size:10px;font-weight:900}.food-bowl{position:relative;width:64px;height:42px;margin:26px auto 10px;border-radius:6px 6px 25px 25px;background:linear-gradient(180deg,#d84849,#ad2e36);box-shadow:0 7px 8px rgba(81,37,25,.2),inset 0 -7px 8px rgba(83,12,22,.14)}
.food-bowl:before{content:"";position:absolute;left:4px;top:-9px;width:56px;height:19px;border:4px solid #d54749;border-radius:50%;background:radial-gradient(circle at 30% 35%,#f1ab36 0 4px,transparent 5px),radial-gradient(circle at 60% 45%,#d57d24 0 5px,transparent 6px),radial-gradient(circle at 78% 30%,#f3c248 0 4px,transparent 5px),#944827}.food-bowl:after{content:"⚡";position:absolute;left:50%;top:10px;transform:translateX(-50%);color:#ffd542;font-size:20px;font-weight:900}.food-count{font-size:9px;font-weight:900}


/* 맞춤 배터리 Plan Selector */
.plan-panel{margin-bottom:8px;padding:11px 10px;background:rgba(255,248,231,.98)}
.plan-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.plan-title{font-size:11px;font-weight:900}.plan-model{padding:4px 7px;border-radius:11px;background:#eaf4df;color:#367b36;font-size:8px;font-weight:900}
.scope-buttons{display:grid;grid-template-columns:1.05fr repeat(5,1fr);gap:5px;margin-bottom:8px}
.scope-btn{min-height:37px;padding:5px 2px;border:1px solid rgba(124,83,43,.16);border-radius:12px;background:#f3e2be;color:#5a412e;font-size:8px;font-weight:900;line-height:1.2;box-shadow:0 3px 7px rgba(69,43,20,.09)}
.scope-btn.active{background:linear-gradient(180deg,#65ae4b,#368e3d);color:#fff;border-color:transparent;box-shadow:0 5px 12px rgba(47,139,58,.25)}
.learn-panel{position:relative;z-index:40;margin-bottom:8px;padding:9px;border-radius:13px;background:linear-gradient(145deg,#fff8df,#f3dfb2);border:1px solid rgba(124,83,43,.13)}.learn-top{display:flex;justify-content:space-between;align-items:center;gap:8px}.learn-title{font-size:10px;font-weight:900}.learn-pill{padding:4px 7px;border-radius:11px;background:#fff;color:#8b6139;font-size:8px;font-weight:900}.learn-desc{margin-top:5px;color:#6f4f38;font-size:8px;line-height:1.45;font-weight:800}.learn-progress{height:8px;margin-top:7px;overflow:hidden;border-radius:12px;background:#dcc79f}.learn-fill{width:0;height:100%;border-radius:inherit;background:linear-gradient(90deg,#62aa49,#ffd44f);transition:width .25s}.learn-status{margin-top:6px;font-size:8px;line-height:1.45;font-weight:900;color:#4b3324}.learn-steps{display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-top:7px}.learn-step{padding:5px 4px;border-radius:9px;background:rgba(255,255,255,.58);color:#806047;font-size:7px;font-weight:900;text-align:center}.learn-step.done{background:#e7f4d9;color:#2f8b3a}.learn-step.active{background:#fff;color:#ef8c32;box-shadow:0 2px 6px rgba(89,56,26,.12)}.learn-btn{position:relative;z-index:80;pointer-events:auto!important;width:100%;min-height:34px;margin-top:7px;border:0;border-radius:11px;background:linear-gradient(90deg,#ef8c32,#ffb24b);color:#fff;font-size:10px;font-weight:900;box-shadow:0 5px 10px rgba(239,140,50,.22)}.learn-btn.ready{background:linear-gradient(90deg,#4a9b42,#75b84e)}.locked-area{opacity:.45;filter:grayscale(.15)}.condition-panel{margin-bottom:8px;padding:8px;border-radius:13px;background:#fff2cf;border:1px solid rgba(124,83,43,.12)}
.condition-title{margin-bottom:7px;font-size:10px;font-weight:900;color:#6f4f38;line-height:1.4}
.condition-row{display:grid;grid-template-columns:.7fr 1fr .7fr 1fr;gap:6px;align-items:center}.condition-row label{font-size:9px;font-weight:900;color:#6c4a2f}.condition-select{width:100%;min-height:34px;border:1px solid rgba(124,83,43,.22);border-radius:10px;background:#fffaf0;color:#4b3324;font-size:10px;font-weight:900;padding:0 7px}.predict-condition-grid{display:grid;grid-template-columns:.72fr 1fr .72fr 1fr;gap:6px;align-items:center;margin-top:8px}.predict-condition-grid label{font-size:9px;font-weight:900;color:#6c4a2f;white-space:nowrap}.condition-help{font-size:9px;line-height:1.5;color:#8a6a45;margin:7px 0 0;font-weight:800}.first-learn-note{padding:8px 9px;border-radius:11px;background:rgba(255,255,255,.58);font-size:10px;line-height:1.55;color:#6f4f38}.predict-btn{width:100%;min-height:36px;margin-top:8px;border:0;border-radius:11px;background:linear-gradient(90deg,#4a9b42,#75b84e);color:#fff;font-size:11px;font-weight:900;box-shadow:0 5px 10px rgba(47,139,58,.2)}.predict-loading{margin-top:6px;min-height:18px;font-size:9px;line-height:1.45;color:#745431;font-weight:800}.predict-loading.active{color:#2f8b3a}
.selected-plan{display:grid;grid-template-columns:1fr .9fr;gap:7px;align-items:stretch}.plan-summary{padding:9px;border-radius:12px;background:#fff4d5;font-size:9px;line-height:1.55;font-weight:800}.plan-summary strong{color:#2f8b3a}.plan-soc{padding:9px;border-radius:12px;background:#f0e0be;text-align:center;font-weight:900}.plan-soc-label{font-size:8px;color:#79583e}.plan-soc-value{margin-top:2px;color:#ef573f;font-size:22px;line-height:1}.plan-soc-sub{margin-top:4px;font-size:8px;color:#76553e;line-height:1.35}.start-clean-primary{width:100%;min-height:46px;margin-top:9px;border:0;border-radius:14px;background:linear-gradient(90deg,#ef8c32,#f2a84d);color:#fff;font-size:13px;font-weight:950;letter-spacing:-.2px;box-shadow:0 7px 15px rgba(239,140,50,.26)}.start-clean-primary:disabled{opacity:.55;filter:grayscale(.12);box-shadow:none}.start-clean-primary small{display:block;margin-top:2px;font-size:9px;font-weight:800;color:rgba(255,255,255,.88)}

/* 공용 패널 헤더/기록 목록 (부품 케어 페이지에서 사용) */
.panel-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}.panel-title{font-size:13px;font-weight:900}.badge{padding:5px 8px;border-radius:12px;background:#fff0ce;color:#805c35;font-size:8px;font-weight:900}
.events{margin-top:9px;padding:15px 14px}.event-item{display:grid;grid-template-columns:43px 1fr;gap:8px;padding:11px 0;border-bottom:1px solid rgba(122,87,51,.12)}.event-item:last-child{border-bottom:0}.event-time{color:#946c43;font-size:9px;font-weight:900}.event-content strong{display:block;margin-bottom:3px;font-size:10px}.event-content span{color:#785a43;font-size:9px;line-height:1.4;font-weight:700}

/* Reward */
.level-panel{padding:17px 15px;text-align:center;background:linear-gradient(145deg,#fff3cc,#ffd98a)}.level-robot{font-size:72px;animation:float 2s ease-in-out infinite}.level-number{margin-top:5px;font-size:25px;font-weight:900}.level-track{height:11px;margin:12px 10px 5px;overflow:hidden;border-radius:10px;background:rgba(126,85,39,.18)}.level-fill{width:55%;height:100%;border-radius:inherit;background:linear-gradient(90deg,#ff7f35,#ffd244)}.level-caption{font-size:9px;font-weight:900}
.reward-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:9px}.reward-card{min-height:125px;padding:14px;border:1px solid rgba(136,87,40,.14);border-radius:16px;background:rgba(255,248,231,.97);box-shadow:var(--shadow);text-align:center}.reward-icon{font-size:38px}.reward-title{margin-top:7px;font-size:11px;font-weight:900}.reward-desc{margin-top:4px;color:#775943;font-size:8px;line-height:1.4;font-weight:800}.reward-btn{width:100%;margin-top:9px;padding:8px;border:0;border-radius:10px;background:#f0dfbc;color:#5c422f;font-size:9px;font-weight:900}

/* Modal */
.modal{position:absolute;z-index:200;inset:0;display:none;align-items:center;justify-content:center;padding:30px;background:rgba(45,33,23,.62);backdrop-filter:blur(4px)}.modal.show{display:flex}.modal-card{width:100%;padding:19px;border-radius:20px;background:#fff8e8;box-shadow:0 18px 45px rgba(28,19,12,.38);animation:popup .18s ease-out}.modal-title{font-size:18px;font-weight:900}.modal-body{margin:13px 0 17px;color:#6c513c;font-size:12px;line-height:1.65;font-weight:700}.modal-actions{display:grid;grid-template-columns:1fr 1fr;gap:9px}.modal-btn{width:100%;padding:11px;border:0;border-radius:12px;font-weight:900}.modal-secondary{background:#efe1c8;color:#5c422f}.modal-primary{background:#ef8c32;color:#fff}.modal-actions.single{grid-template-columns:1fr}.modal-actions.single .modal-secondary{display:none}
.toast{position:absolute;z-index:220;left:50%;bottom:25px;width:max-content;max-width:84%;padding:11px 17px;transform:translateX(-50%) translateY(30px);border-radius:18px;background:rgba(44,37,31,.95);color:#fff;font-size:11px;font-weight:800;opacity:0;pointer-events:none;transition:.25s}.toast.show{opacity:1;transform:translateX(-50%) translateY(0)}

/* Animations */
@keyframes robotIdle{0%,100%{transform:translateX(-50%) translateY(0) rotate(-1deg)}50%{transform:translateX(-50%) translateY(-7px) rotate(1deg)}}
@keyframes robotPatrol{0%{transform:translateX(calc(-50% - 67px)) rotate(-4deg)}50%{transform:translateX(calc(-50% + 67px)) rotate(4deg)}100%{transform:translateX(calc(-50% - 67px)) rotate(-4deg)}}
@keyframes robotCharge{0%,100%{transform:translateX(-50%) scale(1)}50%{transform:translateX(-50%) translateY(-4px) scale(1.04)}}
@keyframes robotLow{0%{transform:translateX(-50%) translateX(-2px)}50%{transform:translateX(-50%) translateX(2px)}100%{transform:translateX(-50%) translateX(-2px)}}
@keyframes robotCelebrate{0%,100%{transform:translateX(-50%) translateY(0)}50%{transform:translateX(-50%) translateY(-35px) rotate(-5deg)}75%{transform:translateX(-50%) translateY(-7px) rotate(5deg)}}
@keyframes robotTap{0%,100%{transform:translateX(-50%) scale(1)}45%{transform:translateX(-50%) scale(.92,1.08)}70%{transform:translateX(-50%) scale(1.08,.94)}}
@keyframes blink{0%,45%,49%,100%{transform:scaleY(1)}47%{transform:scaleY(.08)}}
@keyframes sparkle{0%,100%{transform:scale(.85) rotate(-10deg);opacity:.55}50%{transform:scale(1.18) rotate(8deg);opacity:1}}
@keyframes ringSpin{from{transform:translateX(-50%) rotate(0)}to{transform:translateX(-50%) rotate(360deg)}}
@keyframes dustRise{0%{transform:translateY(0) scale(.6);opacity:0}25%{opacity:.65}100%{transform:translateY(-45px) translateX(15px) scale(1.25);opacity:0}}
@keyframes effectFly{0%{transform:translate(0,0) scale(.7);opacity:0}20%{opacity:1}100%{transform:translate(var(--move-x),-125px) scale(1.35) rotate(var(--rotate));opacity:0}}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-5px)}}
@keyframes popup{from{opacity:0;transform:scale(.88)}to{opacity:1;transform:scale(1)}}
@media(max-width:440px){body{padding:0}.phone{width:100%;height:100vh;border:0;border-radius:0}.notch{display:none}}


/* ===== Sticky robot room: only the lower control panel scrolls ===== */
#homePage.page{
  display:none;
  height:100%;
  padding:0;
  overflow:hidden;
  background:linear-gradient(180deg,#cfaa7d 0%,#e0c39a 39%,#d29c58 40%,#d09850 100%);
}
#homePage.page.active{
  display:flex;
  flex-direction:column;
}
#homePage .room{
  flex:0 0 355px;
  height:355px;
  min-height:355px;
  position:relative;
  z-index:12;
  overflow:hidden;
  box-shadow:0 8px 18px rgba(87,54,25,.14);
}
#homePage .home-dashboard{
  flex:1 1 auto;
  min-height:0;
  overflow-y:auto;
  -webkit-overflow-scrolling:touch;
  padding:10px 8px 24px;
  background:linear-gradient(180deg,rgba(239,205,151,.99),rgba(227,181,110,.99));
  border-top:2px solid rgba(120,76,38,.10);
}
#homePage .home-dashboard::-webkit-scrollbar{width:0;height:0}
#homePage .home-dashboard{scrollbar-width:none;}
@media(max-height:820px){
  #homePage .room{flex-basis:330px;height:330px;min-height:330px;}
}


/* ===== Balanced readable mode: keep original phone shape, enlarge text without stretching layout ===== */
.phone{
  width:min(100%,420px)!important;
  height:960px!important;
  border:8px solid #242321!important;
  border-radius:40px!important;
  overflow:hidden!important;
}
.notch{display:block!important;}
.header{
  height:136px!important;
  padding:24px 14px 8px!important;
}
.pages{height:calc(100% - 136px)!important;}
body{
  padding:8px!important;
  align-items:flex-start!important;
}
body,button,input,select{
  font-size:13px!important;
  word-break:keep-all;
}
.brand{font-size:11px!important;letter-spacing:1.2px!important;}
.app-title{font-size:29px!important;line-height:1.08!important;}
.coin-pill{font-size:14px!important;padding:9px 13px!important;}
.nav{margin-top:11px!important;gap:6px!important;}
.nav-btn{font-size:11px!important;min-height:36px!important;padding:8px 2px!important;border-radius:16px!important;}
.section-kicker{font-size:11px!important;}
.section-title{font-size:24px!important;}
#homePage .room{
  flex:0 0 330px!important;
  height:330px!important;
  min-height:330px!important;
}
#homePage .home-dashboard{
  padding:10px 8px 24px!important;
}
.speech{width:218px!important;min-height:86px!important;font-size:14px!important;line-height:1.55!important;padding:15px 14px!important;}
.speech strong{font-size:17px!important;}
.mode-chip{font-size:11px!important;padding:8px 14px!important;top:111px!important;}
.mission{width:98px!important;padding:9px 8px!important;}
.mission-title{font-size:12px!important;}
.mission-text{font-size:11px!important;line-height:1.45!important;}
.reward-small{font-size:11px!important;}
.quick{right:7px!important;top:86px!important;gap:7px!important;}
.quick-btn{width:54px!important;min-height:53px!important;font-size:10px!important;line-height:1.2!important;}
.quick-btn .icon{font-size:22px!important;}
.actions{gap:4px!important;padding:8px 5px!important;}
.action-btn{font-size:10px!important;line-height:1.15!important;}
.action-icon{font-size:23px!important;}
.plan-panel{padding:12px 10px!important;border-radius:16px!important;}
.plan-head{margin-bottom:8px!important;gap:8px!important;}
.plan-title{font-size:14px!important;line-height:1.25!important;}
.plan-model{font-size:11px!important;padding:5px 8px!important;white-space:nowrap!important;flex:0 0 auto!important;}
.learn-panel{padding:12px 10px!important;border-radius:15px!important;}
.learn-top{align-items:flex-start!important;gap:8px!important;}
.learn-title{font-size:14px!important;line-height:1.3!important;flex:1 1 auto!important;min-width:0!important;}
.learn-pill{
  font-size:11px!important;
  line-height:1.15!important;
  padding:6px 8px!important;
  white-space:nowrap!important;
  word-break:keep-all!important;
  flex:0 0 auto!important;
  min-width:58px!important;
  text-align:center!important;
}
.learn-desc{font-size:12px!important;line-height:1.55!important;}
.learn-status{font-size:12px!important;line-height:1.5!important;}
.learn-step{font-size:11px!important;line-height:1.25!important;padding:7px 5px!important;word-break:keep-all!important;}
.learn-btn{font-size:14px!important;min-height:45px!important;border-radius:13px!important;}
.condition-panel{padding:12px 10px!important;border-radius:15px!important;}
.condition-title{font-size:13px!important;line-height:1.45!important;margin-bottom:9px!important;}
.condition-row,.predict-condition-grid{
  grid-template-columns:76px 1fr 76px 1fr!important;
  gap:8px!important;
  align-items:center!important;
}
.condition-row label,.predict-condition-grid label{
  font-size:12px!important;
  white-space:nowrap!important;
  line-height:1.2!important;
}
.condition-select{font-size:13px!important;min-height:42px!important;padding:0 9px!important;border-radius:11px!important;}
.condition-help{font-size:12px!important;line-height:1.5!important;margin:8px 0 0!important;}
.profile-chip{display:inline-block;background:#edf8df;color:#2f8b3a;border:1px solid #d5ecc3;border-radius:999px;padding:5px 8px!important;font-size:11px!important;font-weight:900;margin-left:4px;white-space:nowrap!important;}
.first-learn-note{font-size:12px!important;line-height:1.55!important;padding:9px 10px!important;}
.predict-btn{font-size:14px!important;min-height:44px!important;border-radius:13px!important;}
.predict-loading{font-size:12px!important;line-height:1.5!important;min-height:24px!important;border-radius:10px;padding:6px 8px;background:rgba(255,255,255,.35);}
.predict-loading.active{background:#eaf4df;color:#2f8b3a;}
.scope-buttons{grid-template-columns:1.05fr repeat(5,1fr)!important;gap:5px!important;}
.scope-btn{font-size:10.5px!important;min-height:42px!important;padding:6px 3px!important;line-height:1.2!important;}
.selected-plan{grid-template-columns:1fr .82fr!important;gap:8px!important;}
.plan-summary{font-size:12px!important;line-height:1.35!important;padding:11px!important;}
.plan-soc{padding:11px 8px!important;}
.plan-soc-label{font-size:11px!important;}
.plan-soc-value{font-size:30px!important;}
.plan-soc-sub{font-size:11px!important;line-height:1.45!important;white-space:normal!important;}
.start-clean-primary{font-size:15px!important;min-height:54px!important;border-radius:15px!important;}
.start-clean-primary small{font-size:11px!important;}
.home-cards{grid-template-columns:1.1fr 1fr .78fr!important;gap:7px!important;}
.mini-card{padding:11px 9px!important;}
.mini-title{font-size:13px!important;}
.battery-info{font-size:11.5px!important;line-height:1.55!important;}
.battery-message{font-size:11px!important;line-height:1.45!important;}
.time-number{font-size:35px!important;}
.time-number small{font-size:14px!important;}
.time-sub{font-size:10.5px!important;}
.time-tip{font-size:11px!important;line-height:1.45!important;}
.food-title,.food-count{font-size:11.5px!important;}
.panel-title{font-size:16px!important;}
.badge{font-size:11px!important;}
.event-content strong{font-size:12.5px!important;}
.event-content span{font-size:11.5px!important;line-height:1.5!important;}
.event-time{font-size:11.5px!important;}
.reward-title{font-size:13px!important;}
.reward-desc{font-size:11.5px!important;line-height:1.5!important;}
.reward-btn{font-size:12px!important;}
.level-number{font-size:30px!important;}
.level-caption{font-size:11px!important;}
.modal{align-items:flex-end!important;justify-content:center!important;padding:0 16px 18px!important;background:rgba(45,33,23,.32)!important;backdrop-filter:blur(2px)!important;}
.modal-card{max-width:392px!important;padding:22px!important;border-radius:24px 24px 20px 20px!important;animation:sheetUp .22s ease-out!important;}
@keyframes sheetUp{from{opacity:0;transform:translateY(46px)}to{opacity:1;transform:translateY(0)}}
.modal-title{font-size:22px!important;}
.modal-body{font-size:15px!important;line-height:1.7!important;}
.modal-btn{font-size:15px!important;padding:13px!important;}
.toast{font-size:13px!important;line-height:1.5!important;min-width:260px;text-align:center;}
@media(max-width:440px){
  body{padding:8px!important;}
  .phone{
    width:min(calc(100% - 16px),420px)!important;
    height:960px!important;
    border:8px solid #242321!important;
    border-radius:40px!important;
  }
  .notch{display:block!important;}
  .header{height:136px!important;}
  .pages{height:calc(100% - 136px)!important;}
  .condition-row,.predict-condition-grid{grid-template-columns:72px 1fr 72px 1fr!important;gap:7px!important;}
  .condition-row label,.predict-condition-grid label{font-size:11.5px!important;}
  .condition-select{font-size:12.5px!important;padding:0 7px!important;}
}
@media(max-height:820px){
  #homePage .room{flex-basis:315px!important;height:315px!important;min-height:315px!important;}
  .speech{top:12px!important;}
}

/* CTA click safety: keep the first-learning button above decorative layers */
#learnPanel{position:relative!important;z-index:60!important;}
#learnBtn{position:relative!important;z-index:9999!important;pointer-events:auto!important;touch-action:manipulation!important;isolation:isolate!important;}
.learn-panel,.plan-panel,.home-dashboard{position:relative!important;}
.learn-panel{z-index:80!important;}
.condition-panel{position:relative!important;z-index:10!important;}

/* ===== Compact learning/profile summary cards ===== */
.compact-note{padding:9px 10px!important;line-height:1.35!important;}
.note-title{font-size:12px;font-weight:950;color:#6f4f38;margin-bottom:7px;}
.profile-mini-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:4px 0 7px;}
.profile-mini-grid span{display:flex;align-items:center;justify-content:center;min-height:28px;border-radius:9px;background:rgba(255,255,255,.72);font-size:11px;font-weight:950;color:#6f4f38;white-space:nowrap;}
.note-caption{font-size:11px;font-weight:850;color:#8a6a45;line-height:1.35;}
.summary-card{display:flex;flex-direction:column;gap:6px;}
.summary-title{font-size:12px;font-weight:950;color:#2f8b3a;margin-bottom:1px;}
.summary-row{display:flex;justify-content:space-between;align-items:center;gap:8px;padding:5px 0;border-bottom:1px dashed rgba(124,83,43,.16);}
.summary-row:last-child{border-bottom:0;}
.summary-key{color:#7c5c3d;font-size:11px;font-weight:900;white-space:nowrap;}
.summary-val{color:#4b3324;font-size:11px;font-weight:950;text-align:right;line-height:1.25;}
.summary-val.em{color:#ef573f;font-size:15px;}
.summary-val.green{color:#2f8b3a;}

/* ===== User guidance banner: persistent next-step explanation ===== */
.flow-guide{
  margin:9px 0 8px;
  padding:10px 11px;
  border:1px solid rgba(78,139,58,.20);
  border-left:5px solid #62aa49;
  border-radius:13px;
  background:linear-gradient(145deg,#f3fbeb,#fff7dc);
  box-shadow:0 4px 10px rgba(79,48,21,.08);
  color:#4b3324;
  font-size:12px;
  line-height:1.5;
  font-weight:850;
}
.flow-guide b{font-weight:1000;color:#2f8b3a;}
.flow-guide .guide-step{display:block;margin-bottom:3px;color:#7a5a3c;font-size:11px;font-weight:1000;}
.flow-guide.warning{border-left-color:#ef8c32;background:linear-gradient(145deg,#fff5dc,#fff0cb);}
.flow-guide.warning b{color:#ef8c32;}
.flow-guide.danger{border-left-color:#ef4e45;background:linear-gradient(145deg,#fff0e9,#fff7dc);}
.flow-guide.danger b{color:#ef4e45;}
.flow-guide.done{border-left-color:#2f8b3a;background:linear-gradient(145deg,#eff9e8,#fff7dc);}
.flow-guide.charging{border-left-color:#f2a84d;background:linear-gradient(145deg,#fff2d2,#fff8e8);}

/* ===== Reward closet: equipped items stay on Roboking ===== */
.robot-accessory{
  position:absolute;
  z-index:34;
  left:50%;
  transform:translateX(-50%);
  pointer-events:none;
  display:none;
  filter:drop-shadow(0 4px 4px rgba(64,38,18,.22));
}
.robot.has-custom-head .crown{display:none!important;}
.robot-head-deco{
  top:-38px;
  min-width:104px;
  height:62px;
  display:none;
  align-items:center;
  justify-content:center;
  text-align:center;
  font-size:43px;
  line-height:1;
}
.robot-head-deco.show{display:flex;animation:decoPop .34s ease-out;}
.robot-head-deco.ribbon{top:-35px;font-size:48px;}
/* 모자는 로보킹 머리 위에 실제로 얹힌 느낌이 나도록 낮게 배치 */
.robot-head-deco.hat{
  top:-38px;
  font-size:66px;
  height:58px;
  transform:translateX(-58%) rotate(-10deg);
  filter:drop-shadow(0 4px 4px rgba(64,38,18,.18));
}
/* 토끼/고양이는 동물 이모지가 아니라 로보킹 자체에 귀가 붙는 장착형 레이어 */
.robot-head-deco.ears{top:-42px;width:138px;height:78px;min-width:138px;}
.robot-head-deco.ears.show{display:block;animation:decoPop .34s ease-out;}
.robo-ear{position:absolute;z-index:2;bottom:4px;filter:drop-shadow(0 3px 3px rgba(64,38,18,.16));}
.robot-head-deco.bunny .robo-ear{
  width:22px;height:62px;border:3px solid #fff;border-radius:16px 16px 12px 12px;
  background:linear-gradient(180deg,#fff 0%,#f1edf8 100%);
}
.robot-head-deco.bunny .robo-ear:after{
  content:"";position:absolute;left:50%;top:8px;width:9px;height:43px;transform:translateX(-50%);
  border-radius:12px;background:linear-gradient(180deg,#ffb5ce,#ffd7e5);
}
.robot-head-deco.bunny .robo-ear.left{left:36px;transform:rotate(-8deg);transform-origin:bottom center;}
.robot-head-deco.bunny .robo-ear.right{right:36px;transform:rotate(8deg);transform-origin:bottom center;}
.robot-head-deco.cat .robo-ear{
  width:36px;
  height:34px;
  bottom:6px;
  background:linear-gradient(180deg,#ffbd4a 0%,#ff922e 82%,#f07725 100%);
  clip-path:polygon(50% 0,4% 100%,96% 100%);
  border-radius:8px;
  filter:drop-shadow(0 3px 3px rgba(64,38,18,.22));
}
.robot-head-deco.cat .robo-ear:after{
  content:"";
  position:absolute;
  left:50%;
  bottom:6px;
  width:16px;
  height:16px;
  transform:translateX(-50%);
  background:linear-gradient(180deg,#ffd6a7,#ff8f80);
  clip-path:polygon(50% 0,8% 100%,92% 100%);
}
.robot-head-deco.cat .robo-ear.left{left:20px;transform:rotate(-16deg);transform-origin:bottom center;}
.robot-head-deco.cat .robo-ear.right{right:20px;transform:rotate(16deg);transform-origin:bottom center;}
.robot-aura-deco{
  position:absolute;
  z-index:5;
  inset:-42px -46px -22px -46px;
  pointer-events:none;
  display:none;
}
.robot-aura-deco.show{display:block;}
.robot-aura-deco span{
  position:absolute;
  font-size:22px;
  filter:drop-shadow(0 3px 3px rgba(64,38,18,.18));
  animation:decoTwinkle 1.6s ease-in-out infinite;
}
.robot-aura-deco span:nth-child(1){left:7px;top:35px;animation-delay:.1s;}
.robot-aura-deco span:nth-child(2){right:2px;top:22px;animation-delay:.45s;}
.robot-aura-deco span:nth-child(3){right:18px;bottom:22px;animation-delay:.8s;}
.robot-aura-deco span:nth-child(4){left:22px;bottom:10px;animation-delay:1.05s;}
.level-robot-preview{position:relative;display:inline-grid;place-items:center;min-width:112px;min-height:94px;margin:0 auto;}
.level-robot-preview .preview-base{font-size:72px;line-height:1;animation:float 2s ease-in-out infinite;}
.preview-head,.preview-aura{position:absolute;pointer-events:none;}
.preview-head{top:-4px;left:50%;transform:translateX(-50%);font-size:35px;filter:drop-shadow(0 3px 3px rgba(64,38,18,.18));}
.preview-head.hat{top:-9px;transform:translateX(-56%) rotate(-10deg);font-size:48px;}
.preview-head.ears{top:-13px;width:92px;height:48px;}
.preview-head.ears .p-ear{position:absolute;bottom:0;filter:drop-shadow(0 2px 2px rgba(64,38,18,.14));}
.preview-head.bunny .p-ear{width:13px;height:42px;border:2px solid #fff;border-radius:12px;background:#f5f1fb;}
.preview-head.bunny .p-ear:after{content:"";position:absolute;left:50%;top:6px;width:5px;height:29px;transform:translateX(-50%);border-radius:8px;background:#ffc2d7;}
.preview-head.bunny .p-ear.left{left:24px;transform:rotate(-8deg)}.preview-head.bunny .p-ear.right{right:24px;transform:rotate(8deg)}
.preview-head.cat .p-ear{width:24px;height:22px;background:linear-gradient(180deg,#ffbd4a,#ff922e 85%);clip-path:polygon(50% 0,4% 100%,96% 100%);}
.preview-head.cat .p-ear:after{content:"";position:absolute;left:50%;bottom:3px;width:10px;height:10px;transform:translateX(-50%);background:linear-gradient(180deg,#ffd6a7,#ff8f80);clip-path:polygon(50% 0,8% 100%,92% 100%);}
.preview-head.cat .p-ear.left{left:13px;transform:rotate(-15deg)}.preview-head.cat .p-ear.right{right:13px;transform:rotate(15deg)}
.preview-aura{inset:0;font-size:18px;animation:decoTwinkle 1.5s ease-in-out infinite;}
.preview-aura .a1{position:absolute;left:2px;top:15px}.preview-aura .a2{position:absolute;right:0;top:28px}.preview-aura .a3{position:absolute;right:12px;bottom:12px}
.reward-btn.equipped{background:linear-gradient(90deg,#4a9b42,#75b84e)!important;color:#fff!important;}
.reward-btn.owned{background:#fff2cf!important;color:#5c422f!important;border:1px solid rgba(124,83,43,.18)!important;}
.reward-card.owned{background:rgba(255,253,240,.98)!important;border-color:rgba(75,155,66,.22)!important;}
.reward-card.equipped{box-shadow:0 0 0 2px rgba(75,155,66,.2), var(--shadow)!important;}
.reward-status{margin-top:5px;color:#4a9b42;font-size:10px;font-weight:950;min-height:13px;}

.reward-folder-tabs{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:10px 0 9px;}
.reward-folder-btn{min-height:43px;border:1px solid rgba(124,83,43,.18);border-radius:14px;background:rgba(255,248,231,.9);color:#6f4f38;font-size:13px;font-weight:950;box-shadow:0 4px 10px rgba(79,48,21,.1);}
.reward-folder-btn.active{background:linear-gradient(90deg,#4a9b42,#75b84e);color:#fff;border-color:transparent;}
.reward-panel{display:block;}
.reward-panel.hidden{display:none;}
.coupon-card{min-height:178px;text-align:left;display:flex;flex-direction:column;align-items:stretch;}
.coupon-card .reward-icon{text-align:center;font-size:34px;line-height:1.05;}
.coupon-card .reward-title{text-align:center;line-height:1.25;min-height:32px;display:flex;align-items:center;justify-content:center;}
.coupon-card .reward-desc{font-size:10.5px;line-height:1.5;text-align:left;min-height:58px;}
.coupon-benefit{margin-top:7px;padding:7px 8px;border-radius:10px;background:#fff2cf;color:#6f4f38;font-size:10px;font-weight:900;line-height:1.35;text-align:left;min-height:42px;display:flex;align-items:center;}
.coupon-card .reward-status{min-height:15px;text-align:center;}
.coupon-card .reward-btn{margin-top:auto;min-height:42px;display:flex;align-items:center;justify-content:center;text-align:center;}
.coupon-card.owned{border-color:rgba(75,155,66,.24);background:rgba(255,253,240,.98);}
.reward-btn.need-coins{background:#efe0bc!important;color:#6f4f38!important;}
.reward-btn.need-coins:after{content:"";}

@keyframes decoPop{from{opacity:0;transform:translateX(-50%) translateY(8px) scale(.7)}to{opacity:1;transform:translateX(-50%) translateY(0) scale(1)}}
@keyframes decoTwinkle{0%,100%{opacity:.45;transform:scale(.88) rotate(-8deg)}50%{opacity:1;transform:scale(1.12) rotate(8deg)}}

/* ===== Room status cleanup: avoid speech overlap + show current robot 배터리 ===== */
#homePage .room .mode-chip{
  top:125px!important;
  left:12px!important;
  right:auto!important;
  bottom:auto!important;
  transform:none!important;
  width:142px!important;
  max-width:142px!important;
  min-height:34px!important;
  padding:7px 9px!important;
  border-radius:15px!important;
  background:rgba(255,248,229,.96)!important;
  box-shadow:0 5px 12px rgba(72,46,23,.18)!important;
  color:#5b3f2d!important;
  font-size:10.5px!important;
  line-height:1.25!important;
  font-weight:950!important;
  text-align:center!important;
  white-space:normal!important;
  word-break:keep-all!important;
}
#homePage .room.cleaning .mode-chip{color:#fff!important;background:rgba(57,143,82,.94)!important;}
#homePage .room.charging .mode-chip{color:#fff!important;background:rgba(242,145,35,.94)!important;}
.robot-soc-badge{
  position:absolute;
  z-index:24;
  right:64px;
  bottom:114px;
  width:98px;
  min-height:44px;
  padding:7px 8px 6px;
  border:2px solid rgba(255,255,255,.82);
  border-radius:16px;
  background:rgba(255,255,255,.95);
  box-shadow:0 7px 16px rgba(60,38,20,.22);
  text-align:center;
  pointer-events:none;
}
.robot-soc-badge span{
  display:block;
  color:#7a5a3c;
  font-size:9.5px;
  line-height:1.05;
  font-weight:950;
  white-space:nowrap;
}
.robot-soc-badge b{
  display:block;
  margin-top:2px;
  color:#2f8b3a;
  font-size:19px;
  line-height:1;
  font-weight:1000;
  letter-spacing:-.5px;
}
.robot-soc-badge.need b{color:#ef8c32;}
.robot-soc-badge.low b{color:#ef4e45;}
.robot-soc-badge.ok b{color:#2f8b3a;}
.robot-soc-badge:after{
  content:"";
  position:absolute;
  left:50%;
  bottom:-8px;
  transform:translateX(-50%);
  border-width:8px 6px 0;
  border-style:solid;
  border-color:rgba(255,255,255,.95) transparent transparent;
}
@media(max-height:820px){
  #homePage .room .mode-chip{top:118px!important;}
  .robot-soc-badge{right:60px;bottom:105px;}
}

/* ===== SVG learned home map card ===== */
.learn-steps.map-ready,
#learnSteps.map-ready{
  display:block!important;
  grid-template-columns:1fr!important;
  gap:0!important;
  width:100%!important;
}
#learnSteps.map-ready .home-map-card{
  width:100%!important;
  max-width:none!important;
}
.home-map-card{
  width:100%;
  padding:6px;
  margin-top:8px;
  border-radius:16px;
  background:rgba(255,255,255,.72);
  border:1px solid rgba(124,83,43,.14);
  box-shadow:inset 0 0 0 1px rgba(255,255,255,.48);
}
.home-map-head{
  display:flex;
  align-items:center;
  justify-content:flex-end;
  gap:8px;
  margin-bottom:5px;
}
.home-map-title{display:none;}
.home-map-badge{
  flex:0 0 auto;
  padding:4px 8px;
  border-radius:999px;
  background:#e8f5dc;
  color:#2f8b3a;
  font-size:10px;
  line-height:1;
  font-weight:950;
  white-space:nowrap;
}
.home-map-img-wrap{
  position:relative;
  overflow:hidden;
  width:100%;
  min-height:210px;
  border-radius:14px;
  background:linear-gradient(145deg,#f6e3ba,#fff7e4);
  border:1px solid rgba(124,83,43,.12);
}
.home-map-svg{display:block;width:100%;height:220px;}
.home-map-caption{display:none;}
.map-room{stroke:#fffdf4;stroke-width:4;filter:drop-shadow(0 3px 4px rgba(67,42,20,.13));}
.map-room.dashed{stroke-dasharray:8 5;stroke:#fffdf4;}
.map-room-label{fill:#64472f;font-size:13px;font-weight:950;text-anchor:middle;dominant-baseline:middle;}
.map-room-sub{fill:#8a6744;font-size:9.4px;font-weight:850;text-anchor:middle;dominant-baseline:middle;}
.map-route{fill:none;stroke:rgba(255,255,255,.82);stroke-width:3;stroke-linecap:round;stroke-dasharray:5 6;}
.map-room-group{cursor:pointer;}
.map-room-group .map-room{transition:opacity .18s ease, filter .18s ease;}
.map-room-group.no-go .map-room{opacity:.42;filter:grayscale(.35);}
.map-no-go-shade{fill:rgba(255,255,255,.48);}
.map-no-go-line{stroke:#8b6f57;stroke-width:4;stroke-linecap:round;opacity:.72;}
.map-action-row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:7px;margin-top:8px;}
.map-action-btn{
  min-height:38px;
  border:0;
  border-radius:13px;
  background:#fff4d8;
  color:#68472d;
  font-size:11.5px;
  font-weight:950;
  box-shadow:inset 0 0 0 1px rgba(135,88,43,.16);
  cursor:pointer;
}
.map-action-btn.active{color:#fff;background:linear-gradient(135deg,#50ae48,#77c75b);box-shadow:0 7px 12px rgba(67,126,56,.20);}
.map-action-btn.danger.active{background:linear-gradient(135deg,#f07a54,#f7a13e);box-shadow:0 7px 12px rgba(190,93,45,.20);}
.map-action-hint{
  margin-top:7px;
  padding:7px 9px;
  border-radius:12px;
  background:rgba(255,250,235,.92);
  border:1px solid rgba(124,83,43,.10);
  color:#6f4f36;
  font-size:11px;
  line-height:1.38;
  font-weight:850;
  text-align:center;
}
.map-action-hint b{color:#2f8b3a;}
.map-legend{
  display:flex;
  align-items:center;
  justify-content:center;
  flex-wrap:wrap;
  gap:7px 10px;
  margin-top:8px;
  padding:8px 8px;
  border-radius:14px;
  background:rgba(255,250,235,.92);
  border:1px solid rgba(124,83,43,.13);
  color:#68472d;
  font-size:12px;
  line-height:1.15;
  font-weight:950;
}
.map-legend-title{color:#4b3324;font-weight:1000;margin-right:2px;}
.map-legend-item{display:inline-flex;align-items:center;gap:5px;}
.map-dot{display:inline-block;width:12px;height:12px;border-radius:999px;border:1px solid rgba(80,50,28,.16);box-shadow:0 1px 2px rgba(72,43,19,.12);}
.dot-clean{background:#cfeec0;}
.dot-normal{background:#ffe08a;}
.dot-dusty{background:#ffb169;}
.dot-focus{background:#ff7d68;}

/* ===== Main map CTA row ===== */
.learn-actions{display:grid;grid-template-columns:1fr;gap:8px;margin-top:8px;}
.learn-actions.ready{grid-template-columns:1fr 1fr;}
.clean-execute-btn{
  position:relative;
  z-index:9999;
  width:100%;
  min-height:45px;
  border:0;
  border-radius:13px;
  background:linear-gradient(90deg,#41a346,#79c75a);
  color:#fff;
  font-size:14px;
  font-weight:950;
  box-shadow:0 7px 12px rgba(67,126,56,.22);
  cursor:pointer;
  pointer-events:auto!important;
  touch-action:manipulation!important;
}
.clean-execute-btn:disabled{opacity:.55;cursor:not-allowed;filter:grayscale(.12);}
.condition-panel.manual-mode{margin-top:9px!important;background:linear-gradient(145deg,#fff8e6,#f3dfb2)!important;}
.condition-panel.manual-mode .condition-title:before{content:"✍️ ";}

/* ===== Home simplification: one main clean-prep button, no extra lower cards ===== */
.scope-buttons,
.selected-plan,
.start-clean-primary,
.actions,
.home-cards{display:none!important;}
.condition-panel{margin-bottom:0!important;}
.predict-btn{position:relative;z-index:30;}
#flowGuide{margin-top:9px;}

.manual-action-row{
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:8px;
  margin-top:10px;
}
.manual-clean-btn{
  position:relative;
  z-index:31;
  width:100%;
  min-height:43px;
  border:0;
  border-radius:13px;
  color:#fff;
  background:linear-gradient(90deg,#f69028,#f9b047);
  font-size:12.5px;
  font-weight:950;
  box-shadow:0 7px 12px rgba(210,117,35,.18);
  cursor:pointer;
}
.manual-clean-btn.ready{
  background:linear-gradient(90deg,#41a346,#79c75a);
  box-shadow:0 7px 12px rgba(67,126,56,.20);
}
.manual-clean-btn:disabled{
  opacity:.55;
  cursor:not-allowed;
  filter:grayscale(.12);
}
.manual-action-row .predict-btn{
  margin-top:0!important;
}
.manual-combo-btn{
  width:100%;
  min-height:46px!important;
  margin-top:10px!important;
  border-radius:14px!important;
  font-size:13px!important;
  font-weight:950!important;
  background:linear-gradient(90deg,#41a346,#79c75a)!important;
  color:#fff!important;
  box-shadow:0 7px 12px rgba(67,126,56,.22)!important;
}
.manual-combo-btn.running{
  background:linear-gradient(90deg,#f69028,#f9b047)!important;
}

/* ===== Home WOW features: station motion, route, map progress, prep cards ===== */
.room:after{
  content:"";
  position:absolute;
  z-index:6;
  left:92px;
  bottom:92px;
  width:54px;
  height:15px;
  border-radius:999px;
  background:rgba(64,42,26,.18);
  opacity:0;
  transform:scale(.7);
  transition:.25s ease;
}
.room.returning:after,
.room.docked:after{
  opacity:1;
  transform:scale(1);
}
.room.returning .robot{
  animation:robotReturnStation .95s ease-in-out forwards!important;
}
.room.docked .robot,
.room.charging .robot{
  animation:robotDockBreath 1.15s ease-in-out infinite!important;
  transform:translateX(calc(-50% - 105px)) translateY(-38px) scale(.72)!important;
}
.room.departing .robot{
  animation:robotDepartStation .85s ease-in-out forwards!important;
}
.room.charging .charge-ring,
.room.docked .charge-ring{
  left:104px!important;
  bottom:69px!important;
  width:118px!important;
  height:74px!important;
  opacity:.92!important;
  animation:ringSpin 1.05s linear infinite!important;
}
.room.cleaning .clean-path,
.room.route-preview .clean-path{
  opacity:1;
}
.room.route-preview .clean-fill{
  width:64%!important;
  animation:pathPreviewPulse 1.7s ease-in-out infinite;
}
.room.docked .clean-path,
.room.returning .clean-path{
  opacity:.28;
}
.map-room-group{cursor:pointer;}
.map-room-group .map-room{
  transition:opacity .18s ease, filter .18s ease, stroke-width .18s ease;
}
.map-room-group.planned .map-room{
  stroke:#36b04a;
  stroke-width:5.2;
  filter:drop-shadow(0 0 3px rgba(54,176,74,.35));
  animation:plannedGreenBlink 1.15s ease-in-out infinite;
}
.map-room-group.dirty-selected .map-room{
  stroke:#36b04a;
  stroke-width:5.2;
  filter:drop-shadow(0 0 3px rgba(54,176,74,.35));
  animation:plannedGreenBlink 1.05s ease-in-out infinite;
}
.map-dirty-ring,
.map-dirty-spark{
  display:none!important;
}
.map-room-group.dimmed .map-room{
  opacity:.28;
  filter:grayscale(.35);
}
.map-room-group.cleaning-zone .map-room{
  animation:plannedGreenBlink .9s ease-in-out infinite;
  stroke:#27a844;
  stroke-width:6.2;
}
.map-room-group.completed .map-room{
  opacity:.62;
}
.map-room-group.no-go .map-room{
  opacity:.40;
  filter:grayscale(.42);
}
.map-no-go-shade{fill:rgba(255,255,255,.52);}
.map-no-go-line{stroke:#6b5543;stroke-width:4.5;stroke-linecap:round;opacity:.78;}
.map-check{
  fill:#2f8b3a;
  font-size:14px;
  font-weight:950;
  text-anchor:middle;
  dominant-baseline:middle;
}
.map-route,
.map-route.active-route{
  opacity:0!important;
  display:none!important;
}
.map-action-hint.status-ready{
  border-color:rgba(73,163,68,.22);
  background:#f3ffe9;
}
.map-recommend-card{
  display:flex;
  align-items:center;
  gap:7px;
  margin-top:7px;
  padding:8px 9px;
  border-radius:13px;
  background:linear-gradient(135deg,#fffaf0,#f9e5b9);
  border:1px solid rgba(124,83,43,.12);
  color:#60452f;
  font-size:11px;
  line-height:1.38;
  font-weight:900;
}
.map-recommend-card .rec-icon{
  flex:0 0 auto;
  display:grid;
  place-items:center;
  width:26px;
  height:26px;
  border-radius:50%;
  background:#fff4d8;
  box-shadow:inset 0 0 0 1px rgba(124,83,43,.13);
}
.map-prep-card{
  display:grid;
  grid-template-columns:1fr auto;
  gap:6px;
  align-items:center;
  margin-top:7px;
  padding:8px 10px;
  border-radius:13px;
  background:rgba(255,255,255,.74);
  border:1px solid rgba(124,83,43,.12);
}
.map-prep-title{
  color:#4b3324;
  font-size:11.5px;
  line-height:1.35;
  font-weight:950;
}
.map-prep-sub{
  margin-top:2px;
  color:#7b5a3e;
  font-size:10px;
  line-height:1.35;
  font-weight:850;
}
.map-prep-badge{
  padding:5px 8px;
  border-radius:999px;
  color:#fff;
  background:#ef8c32;
  font-size:10px;
  font-weight:950;
  white-space:nowrap;
}
@keyframes robotReturnStation{
  0%{transform:translateX(-50%) translateY(0) scale(1) rotate(0)}
  55%{transform:translateX(calc(-50% - 72px)) translateY(-14px) scale(.88) rotate(-4deg)}
  100%{transform:translateX(calc(-50% - 105px)) translateY(-38px) scale(.72) rotate(0)}
}
@keyframes robotDepartStation{
  0%{transform:translateX(calc(-50% - 105px)) translateY(-38px) scale(.72)}
  65%{transform:translateX(calc(-50% - 40px)) translateY(-12px) scale(.9) rotate(3deg)}
  100%{transform:translateX(-50%) translateY(0) scale(1)}
}
@keyframes robotDockBreath{
  0%,100%{transform:translateX(calc(-50% - 105px)) translateY(-38px) scale(.72)}
  50%{transform:translateX(calc(-50% - 105px)) translateY(-43px) scale(.75)}
}
@keyframes pathPreviewPulse{
  0%,100%{opacity:.55}
  50%{opacity:1}
}
@keyframes mapZonePulse{
  0%,100%{filter:drop-shadow(0 4px 6px rgba(55,164,71,.22))}
  50%{filter:drop-shadow(0 4px 10px rgba(55,164,71,.58))}
}
@keyframes routeDash{
  from{stroke-dashoffset:36}
  to{stroke-dashoffset:0}
}
@keyframes dirtyZoneGlow{
  0%,100%{filter:drop-shadow(0 0 3px rgba(255,216,77,.34))}
  50%{filter:drop-shadow(0 0 8px rgba(255,216,77,.74))}
}
@keyframes dirtyRingDash{
  from{stroke-dashoffset:24}
  to{stroke-dashoffset:0}
}
@keyframes plannedGreenBlink{
  0%,100%{
    stroke:#36b04a;
    stroke-width:4.2;
    filter:drop-shadow(0 0 2px rgba(54,176,74,.25));
  }
  50%{
    stroke:#20c949;
    stroke-width:6.4;
    filter:drop-shadow(0 0 6px rgba(54,176,74,.55));
  }
}


/* ===== Map robot marker: 지도 위에서 로보킹이 실제로 움직이며 청소하는 표현 ===== */
.map-sweep{fill:none;stroke:rgba(255,255,255,.80);stroke-width:7;stroke-linecap:round;stroke-linejoin:round;pointer-events:none;}
.map-robot{pointer-events:none;}
.map-robot-shadow{fill:rgba(64,42,26,.22);}
.map-robot-shell{fill:#fbfbf8;stroke:#a29b92;stroke-width:1.4;}
.map-robot-face{fill:#1f2122;}
.map-robot-eye{fill:#fff;}
.map-robot-light{fill:#62aa49;animation:mapRobotBlink 1s ease-in-out infinite;}
.map-robot-body{animation:mapRobotBob .7s ease-in-out infinite;}
.map-robot-crown{font-size:7px;text-anchor:middle;}
.map-robot-puff{fill:rgba(255,255,255,.75);animation:mapPuff 1s ease-out infinite;}
.map-robot-puff.p2{animation-delay:.33s;}
.map-robot-puff.p3{animation-delay:.66s;}
@keyframes mapRobotBob{0%,100%{transform:translateY(0)}50%{transform:translateY(-1.2px)}}
@keyframes mapRobotBlink{0%,100%{opacity:.45}50%{opacity:1}}
@keyframes mapPuff{0%{opacity:.8;transform:translate(0,0) scale(.5)}100%{opacity:0;transform:translate(-9px,-4px) scale(1.4)}}

/* ===== Top learning / clean action buttons alignment ===== */
.learn-actions.ready{
  grid-template-columns:1fr 1fr!important;
  align-items:stretch!important;
}
.learn-actions.ready .learn-btn,
.learn-actions.ready .clean-execute-btn{
  width:100%!important;
  height:48px!important;
  min-height:48px!important;
  padding:0 8px!important;
  border-radius:14px!important;
  display:flex!important;
  align-items:center!important;
  justify-content:center!important;
  text-align:center!important;
  line-height:1.18!important;
  white-space:normal!important;
  word-break:keep-all!important;
  box-sizing:border-box!important;
  margin:0!important;
}
.learn-actions.ready .learn-btn{
  background:linear-gradient(90deg,#41a346,#79c75a)!important;
  color:#fff!important;
  font-size:12.5px!important;
  font-weight:950!important;
  box-shadow:0 7px 12px rgba(67,126,56,.20)!important;
}
.learn-actions.ready .clean-execute-btn{
  background:linear-gradient(90deg,#f69028,#f9b047)!important;
  color:#fff!important;
  font-size:13px!important;
  font-weight:950!important;
  box-shadow:0 7px 12px rgba(210,117,35,.22)!important;
}
.learn-actions.ready .clean-execute-btn:disabled{
  opacity:.62!important;
  filter:grayscale(.08)!important;
}


/* ============================================================
   NEW PAGE 2 · 부품 케어 (부품 상태 + 실시간 케어 기록)
   ============================================================ */
.parts-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;}
.part-card{display:flex;align-items:center;gap:8px;padding:12px 10px;border:1px solid rgba(136,87,40,.14);border-radius:16px;background:rgba(255,248,231,.97);box-shadow:var(--shadow);cursor:pointer;text-align:left;}
.part-card:active{transform:scale(.98);}
.part-icon{flex:0 0 auto;width:44px;height:44px;display:grid;place-items:center;border-radius:14px;background:#f5eddb;font-size:25px;}
.part-info{flex:1 1 auto;min-width:0;}
.part-name{font-size:11.5px;font-weight:900;color:#7a5a3c;}
.part-status{margin-top:3px;font-size:12.5px;font-weight:950;line-height:1.3;word-break:keep-all;}
.part-face{flex:0 0 auto;width:30px;height:30px;border-radius:50%;display:grid;place-items:center;font-size:17px;}
.part-card.good .part-status{color:#2f8b3a;}.part-card.good .part-face{background:#dff3cd;}
.part-card.check .part-status{color:#e07a1f;}.part-card.check .part-face{background:#ffe6c2;}
.part-card.bad .part-status{color:#ef4e45;}.part-card.bad .part-face{background:#ffd9d4;}
.care-summary{margin-top:9px;padding:14px;}
.care-lead{font-size:12px;line-height:1.55;font-weight:850;color:#6f4f38;}
.care-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin-top:10px;}
.care-stat{padding:10px 6px;border-radius:13px;background:#fff2cf;text-align:center;}
.care-stat span{display:block;font-size:10.5px;font-weight:900;color:#7a5a3c;line-height:1.3;}
.care-stat b{display:block;margin-top:5px;color:#2f8b3a;font-size:22px;font-weight:1000;line-height:1;}
.care-stat small{font-size:10px;font-weight:900;color:#7a5a3c;}
.care-health-row{display:flex;align-items:center;gap:8px;margin-top:12px;font-size:11.5px;font-weight:900;}
.care-health-track{flex:1;height:10px;overflow:hidden;border-radius:10px;background:#ead9b9;}
.care-health-fill{height:100%;border-radius:inherit;background:linear-gradient(90deg,#62aa49,#ffd44f);transition:width .3s;}
.care-note{margin-top:10px;padding:9px 10px;border-radius:12px;background:#eaf4df;color:#2f8b3a;font-size:11.5px;line-height:1.5;font-weight:850;}
.event-tag{display:inline-block;margin-left:5px;padding:2px 7px;border-radius:999px;background:#eaf4df;color:#2f8b3a;font-size:9.5px!important;font-weight:950;vertical-align:middle;line-height:1.3;}
.modal-emoji{font-size:64px;text-align:center;padding:10px 0 14px;}
.modal-img{width:100%;max-height:300px;object-fit:cover;border-radius:14px;margin-bottom:10px;display:block;}

/* ============================================================
   NEW PAGE 3 · 예약 청소 (출퇴근 맞춤 + 테마 기간 청소)
   ============================================================ */
.sched-panel{padding:14px;margin-bottom:9px;}
.sched-head{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;}
.sched-title{font-size:15px;font-weight:950;}
.sched-desc{margin-top:4px;font-size:11.5px;line-height:1.5;font-weight:800;color:#76533b;}
.switch{flex:0 0 auto;position:relative;width:50px;height:28px;border:0;border-radius:999px;background:#d8c7a6;transition:.2s;}
.switch .knob{position:absolute;top:3px;left:3px;width:22px;height:22px;border-radius:50%;background:#fff;box-shadow:0 2px 5px rgba(0,0,0,.2);transition:.2s;}
.switch.on{background:#4a9b42;}.switch.on .knob{left:25px;}
.sched-body{margin-top:11px;}
.sched-body.off{opacity:.5;pointer-events:none;filter:grayscale(.2);}
.time-row{display:grid;grid-template-columns:38px 1fr 38px 1fr;gap:7px;align-items:center;}
.time-row label{font-size:12px;font-weight:900;color:#6c4a2f;}
.day-chips{display:grid;grid-template-columns:repeat(7,1fr);gap:5px;margin-top:9px;}
.day-chip{min-height:34px;border:1px solid rgba(124,83,43,.18);border-radius:11px;background:#f3e2be;color:#6f4f38;font-size:12px;font-weight:950;}
.day-chip.on{background:linear-gradient(180deg,#65ae4b,#368e3d);color:#fff;border-color:transparent;}
.sched-options{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:9px;}
.sched-opt{min-height:40px;border:1px solid rgba(124,83,43,.16);border-radius:12px;background:#fff4d5;color:#5a412e;font-size:11.5px;font-weight:950;line-height:1.25;padding:6px;}
.sched-opt.active{background:linear-gradient(90deg,#ef8c32,#ffb24b);color:#fff;border-color:transparent;}
.sched-preview{margin-top:10px;padding:10px 11px;border-radius:12px;background:#eaf4df;color:#2f8b3a;font-size:12px;line-height:1.6;font-weight:900;}
.sched-preview b{color:#ef573f;}
.sched-preview.off{background:#f3e2be;color:#7a5a3c;}
.sub-title{margin:14px 0 8px;font-size:15px;font-weight:950;}
.theme-list{display:flex;flex-direction:column;gap:8px;}
.theme-card{display:grid;grid-template-columns:46px 1fr auto;gap:10px;align-items:center;padding:12px;border:1px solid rgba(136,87,40,.14);border-radius:16px;background:rgba(255,248,231,.97);box-shadow:var(--shadow);}
.theme-card.on{border-color:rgba(75,155,66,.35);box-shadow:0 0 0 2px rgba(75,155,66,.18),var(--shadow);}
.theme-card.past{opacity:.6;}
.theme-icon{width:46px;height:46px;display:grid;place-items:center;border-radius:14px;background:#fff2cf;font-size:25px;}
.theme-info{min-width:0;}
.theme-name{font-size:13.5px;font-weight:950;line-height:1.25;}
.theme-period{margin-top:2px;font-size:11px;font-weight:900;color:#946c43;}
.theme-desc{margin-top:4px;font-size:11.5px;line-height:1.45;font-weight:800;color:#6f4f38;}
.theme-chips{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px;}
.theme-chips span{padding:3px 7px;border-radius:999px;background:#f3e2be;font-size:10px;font-weight:950;color:#6f4f38;}
.theme-state{display:inline-block;margin-left:5px;padding:2px 7px;border-radius:999px;font-size:9.5px;font-weight:950;background:#fff0ce;color:#805c35;vertical-align:middle;}
.theme-state.live{background:#eaf4df;color:#2f8b3a;}
.theme-state.past{background:#eee6da;color:#8a7a68;}
.theme-btn{min-width:62px;min-height:40px;border:0;border-radius:12px;background:#f0dfbc;color:#5c422f;font-size:11.5px;font-weight:950;}
.theme-btn.on{background:linear-gradient(90deg,#4a9b42,#75b84e);color:#fff;}
.upcoming{margin-top:9px;padding:14px;}
.upcoming-item{display:grid;grid-template-columns:34px 1fr;gap:8px;align-items:center;padding:9px 0;border-bottom:1px dashed rgba(122,87,51,.16);}
.upcoming-item:last-child{border-bottom:0;}
.upcoming-icon{font-size:22px;text-align:center;}
.upcoming-item strong{display:block;font-size:12.5px;}
.upcoming-item span{display:block;margin-top:2px;font-size:11px;color:#785a43;font-weight:800;line-height:1.4;}
.upcoming-empty{padding:6px 0;font-size:12px;color:#8a6a45;font-weight:850;line-height:1.5;}

/* ============================================================
   NEW PAGE 4 · 이벤트 (오늘의 발견 / 미션 / 사진첩)
   ============================================================ */
.event-tabs{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin:0 0 9px;}
.event-tabs .reward-folder-btn{font-size:12px!important;padding:0 4px;white-space:nowrap;}
.found-top{display:grid;grid-template-columns:1fr 1fr;gap:8px;}
.found-card{padding:11px 10px;border:1px solid rgba(136,87,40,.14);border-radius:16px;background:rgba(255,248,231,.97);box-shadow:var(--shadow);display:flex;flex-direction:column;}
.found-card-title{display:flex;align-items:center;gap:5px;font-size:12.5px;font-weight:950;margin-bottom:8px;}
.found-photo{position:relative;height:92px;border-radius:12px;overflow:hidden;background:linear-gradient(145deg,#e9cfa8,#f7e6c9);display:grid;place-items:center;font-size:44px;}
.found-photo img{width:100%;height:100%;object-fit:cover;display:block;}
.found-name{margin-top:8px;font-size:13px;font-weight:950;color:#e07a1f;}
.found-name.done{color:#2f8b3a;}
.found-desc{margin-top:3px;font-size:11px;line-height:1.45;font-weight:800;color:#6f4f38;}
.found-meta{margin-top:5px;font-size:10.5px;font-weight:900;color:#7a5a3c;line-height:1.5;}
.found-btn{margin-top:auto;padding-top:8px;}
.found-btn button{width:100%;min-height:34px;border:0;border-radius:11px;background:#f0dfbc;color:#5c422f;font-size:11.5px;font-weight:950;}
.found-map{flex:1;min-height:150px;border-radius:12px;overflow:hidden;background:#fbf1de;border:1px solid rgba(124,83,43,.12);}
.found-map svg{width:100%;height:100%;display:block;}
.fm-room{fill:#f6e9d2;stroke:#c9ad82;stroke-width:2;}
.fm-label{fill:#7a5a3c;font-size:11px;font-weight:900;text-anchor:middle;}
.fm-pin{animation:float 1.6s ease-in-out infinite;}
.found-list{margin-top:9px;padding:13px 12px;}
.found-item{display:grid;grid-template-columns:48px 1fr auto;gap:9px;align-items:center;padding:9px 0;border-bottom:1px dashed rgba(122,87,51,.16);cursor:pointer;}
.found-item:last-child{border-bottom:0;}
.found-thumb{width:48px;height:48px;border-radius:12px;overflow:hidden;display:grid;place-items:center;background:#f5eddb;font-size:26px;}
.found-thumb img{width:100%;height:100%;object-fit:cover;display:block;}
.found-item strong{display:block;font-size:12.5px;}
.found-item span{display:block;margin-top:2px;font-size:11px;color:#785a43;font-weight:800;}
.found-right{text-align:right;font-size:10.5px;color:#7a5a3c;font-weight:900;line-height:1.55;white-space:nowrap;}
.found-item.done{opacity:.55;}
.mission-summary{padding:13px 14px;margin-bottom:9px;background:linear-gradient(145deg,#fff3cc,#ffd98a);display:flex;justify-content:space-between;align-items:center;gap:10px;}
.mission-summary .ms-title{font-size:13.5px;font-weight:950;}
.mission-summary .ms-desc{margin-top:3px;font-size:11px;font-weight:850;color:#76533b;line-height:1.45;}
.mission-summary .ms-count{flex:0 0 auto;text-align:center;padding:8px 12px;border-radius:12px;background:rgba(255,255,255,.7);font-size:10.5px;font-weight:950;color:#7a5a3c;}
.mission-summary .ms-count b{display:block;font-size:22px;color:#ef573f;line-height:1;margin-bottom:2px;}
.mission-card{padding:12px;margin-bottom:8px;border:1px solid rgba(136,87,40,.14);border-radius:16px;background:rgba(255,248,231,.97);box-shadow:var(--shadow);}
.mission-head{display:flex;align-items:center;gap:8px;}
.mission-head .m-icon{font-size:24px;}
.mission-head .m-name{font-size:13.5px;font-weight:950;flex:1;}
.mission-head .m-count{font-size:11px;font-weight:900;color:#7a5a3c;white-space:nowrap;}
.tier-row{display:grid;grid-template-columns:30px 1fr auto;gap:8px;align-items:center;margin-top:9px;}
.tier-medal{font-size:22px;text-align:center;filter:grayscale(1);opacity:.45;}
.tier-row.reached .tier-medal{filter:none;opacity:1;}
.tier-info{min-width:0;}
.tier-goal{font-size:11.5px;font-weight:950;}
.tier-track{height:7px;margin-top:4px;border-radius:10px;overflow:hidden;background:#ead9b9;}
.tier-fill{height:100%;border-radius:inherit;background:linear-gradient(90deg,#ff8e33,#ffca43);transition:width .3s;}
.tier-btn{min-width:78px;min-height:34px;border:0;border-radius:11px;background:#f0dfbc;color:#8a7a68;font-size:11px!important;font-weight:950;padding:0 8px;}
.tier-btn.claim{background:linear-gradient(90deg,#4a9b42,#75b84e);color:#fff;animation:pulseBtn 1.4s ease-in-out infinite;}
.tier-btn.claimed{background:#eaf4df;color:#2f8b3a;}
@keyframes pulseBtn{0%,100%{transform:scale(1)}50%{transform:scale(1.05)}}
.photo-info{padding:12px 13px;margin-bottom:9px;display:flex;justify-content:space-between;align-items:center;gap:8px;}
.photo-info .p-title{font-size:13.5px;font-weight:950;}
.photo-info .p-desc{margin-top:3px;font-size:11px;font-weight:800;color:#76533b;line-height:1.45;}
.photo-info .p-count{flex:0 0 auto;text-align:center;padding:8px 10px;border-radius:12px;background:#fff2cf;font-size:10.5px;font-weight:950;color:#7a5a3c;}
.photo-info .p-count b{display:block;font-size:20px;color:#ef573f;line-height:1;margin-bottom:2px;}
.photo-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;}
.photo-tile{position:relative;border-radius:16px;overflow:hidden;background:#f5eddb;box-shadow:var(--shadow);cursor:pointer;aspect-ratio:1/1;}
.photo-tile img{width:100%;height:100%;object-fit:cover;display:block;}
.photo-tile .ph-emoji{width:100%;height:100%;display:grid;place-items:center;font-size:56px;background:linear-gradient(145deg,#f6e3ba,#fff7e4);}
.photo-cap{position:absolute;left:0;right:0;bottom:0;padding:16px 9px 8px;background:linear-gradient(180deg,transparent,rgba(45,33,23,.74));color:#fff;font-size:11px;font-weight:950;line-height:1.3;}
.photo-cap small{display:block;font-weight:800;opacity:.85;font-size:9.5px;}
.photo-empty{padding:14px;margin-top:9px;text-align:center;font-size:11.5px;line-height:1.6;font-weight:850;color:#6f4f38;}
.photo-empty code{background:#fff2cf;padding:2px 6px;border-radius:6px;font-size:11px;}
.nav-dot{display:inline-block;width:7px;height:7px;margin-left:3px;border-radius:50%;background:#ffd44f;vertical-align:middle;box-shadow:0 0 0 2px rgba(255,212,79,.25);}

/* ===== Home information architecture: Map / AI preparation / Manual cleaning ===== */
.home-section{
  margin-bottom:10px;
  padding:13px 11px;
  overflow:hidden;
}
.home-section-head{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:10px;
  margin-bottom:10px;
}
.home-section-kicker{
  margin-bottom:3px;
  color:#a27145;
  font-size:9px;
  line-height:1;
  font-weight:950;
  letter-spacing:1.1px;
}
.home-section-title{
  color:#4b3324;
  font-size:16px;
  line-height:1.25;
  font-weight:1000;
}
.home-section-badge{
  flex:0 0 auto;
  padding:6px 9px;
  border-radius:999px;
  background:#f3e4c5;
  color:#76553e;
  font-size:10.5px;
  line-height:1;
  font-weight:950;
}
.home-section-badge.ready{background:#e7f4d9;color:#2f8b3a;}
.home-section-badge.learning{background:#fff0ca;color:#e07c22;}
.map-section{background:rgba(255,248,231,.98);}
.map-section-content{width:100%;}
.map-empty-state{
  min-height:138px;
  display:flex;
  flex-direction:column;
  align-items:center;
  justify-content:center;
  gap:6px;
  padding:17px 12px;
  border:1px dashed rgba(124,83,43,.24);
  border-radius:15px;
  background:linear-gradient(145deg,#fff8e6,#f4e3bf);
  color:#7a5a3c;
  text-align:center;
}
.map-empty-state.learning{border-style:solid;background:linear-gradient(145deg,#fff5d9,#f5dfae);}
.map-empty-icon{font-size:31px;line-height:1;}
.map-empty-state b{color:#4b3324;font-size:13px;font-weight:1000;}
.map-empty-state span:last-child{font-size:11px;line-height:1.5;font-weight:850;}
.map-section .home-map-card{
  margin:0;
  padding:0;
  border:0;
  border-radius:0;
  background:transparent;
  box-shadow:none;
}
.map-section .home-map-head{margin:0 0 7px;}
.room .mode-chip{display:none!important;}
.prep-section{margin-bottom:10px!important;}
.prep-section .plan-head{align-items:center;}
.prep-section .learn-panel{margin-bottom:0;}
.direct-clean-section{
  margin:0!important;
  padding:13px 11px!important;
  border-radius:17px!important;
  background:rgba(255,248,231,.98)!important;
  box-shadow:var(--shadow);
}
.direct-clean-section .home-section-kicker{margin-bottom:4px;}
.direct-clean-section .condition-title{
  margin-bottom:9px!important;
  color:#4b3324!important;
  font-size:16px!important;
  font-weight:1000!important;
}
.direct-clean-section.condition-panel.manual-mode{margin-top:0!important;}
.direct-clean-section.condition-panel.manual-mode .condition-title:before{content:none!important;}


/* ===== Home visual adjustment: bigger station + mission moved right ===== */
#homePage .house{
  left:76px!important;
  top:112px!important;
  width:94px!important;
  height:80px!important;
  border-radius:38px 38px 9px 9px!important;
  box-shadow:0 8px 14px rgba(54,36,23,.24)!important;
}
#homePage .house:before{
  left:27px!important;
  bottom:0!important;
  width:40px!important;
  height:44px!important;
  border-radius:20px 20px 0 0!important;
}
#homePage .house:after{
  top:13px!important;
  left:27px!important;
  font-size:8.5px!important;
  letter-spacing:.2px!important;
}
#homePage .mission{
  left:auto!important;
  right:10px!important;
  bottom:14px!important;
  width:96px!important;
  z-index:18!important;
}

</style>
</head>

<body>
<div class="phone">
  <div class="notch"></div>
  <div class="screen">

    <header class="header">
      <div class="header-top">
        <div>
          <div class="brand">LG ROBO CARE</div>
          <div class="app-title">로보킹 키우기</div>
        </div>
        <div class="coin-pill">🪙 <span id="coinText">050</span></div>
      </div>
      <nav class="nav">
        <button class="nav-btn active" data-page="homePage">홈</button>
        <button class="nav-btn" data-page="batteryPage">부품케어</button>
        <button class="nav-btn" data-page="recordPage">예약청소</button>
        <button class="nav-btn" data-page="eventPage">이벤트<span class="nav-dot" id="eventNavDot" style="display:none"></span></button>
        <button class="nav-btn" data-page="rewardPage">리워드</button>
      </nav>
    </header>

    <main class="pages">

      <section class="page active" id="homePage">
        <div class="room" id="room">
          <div class="wall-light"></div><div class="floor"></div>
          <div class="plant">🪴</div><div class="house"></div><div class="sofa"></div>
          <div class="speech" id="speech"><strong>배가 든든해요!</strong><br>청소를 준비할게요!</div>
          <div class="mode-chip" id="modeChip">✨ 로보킹 맞춤 준비</div>
          <div class="rug"></div>
          <div class="clean-path"><div class="clean-fill" id="cleanFill"></div></div>
          <div class="charge-ring"></div>
          <div class="dust"><span></span><span></span><span></span><span></span><span></span><span></span></div>

          <div class="robot" id="robot" data-action="pet">
            <div class="robot-aura-deco" id="robotAuraDeco"><span>✨</span><span>✨</span><span>✨</span><span>✨</span></div>
            <div class="crown">👑</div>
            <div class="robot-accessory robot-head-deco" id="robotHeadDeco"></div>
            <div class="spark" id="spark">✨</div><div class="robot-top"></div>
            <div class="face">
              <div class="eye left"></div><div class="eye right"></div>
              <div class="cheek left"></div><div class="cheek right"></div><div class="mouth"></div>
            </div>
            <div class="robot-accessory robot-body-deco" id="robotBodyDeco"></div>
            <div class="slot"></div>
          </div>
          <div class="robot-soc-badge" id="robotSocBadge"><span>🔋 현재 배터리</span><b>20%</b></div>

          <div class="mission">
            <div class="mission-title">오늘의 미션</div>
            <div class="mission-text">거실 청소 1회 완료하기</div>
            <div class="mission-progress">
              <div class="mission-track"><div class="mission-fill" id="missionFill"></div></div>
              <span class="reward-small">+50</span>
            </div>
          </div>

          <div class="effect-layer" id="effectLayer"></div>
        </div>

        <div class="home-dashboard">

          <section class="panel home-section map-section" aria-labelledby="mapSectionTitle">
            <div class="home-section-head">
              <div>
                <div class="home-section-kicker">CLEANING MAP</div>
                <div class="home-section-title" id="mapSectionTitle">🗺️ 맵</div>
              </div>
              <div class="home-section-badge" id="mapSectionBadge">학습 전</div>
            </div>
            <div class="map-section-content" id="mapSectionContent">
              <div class="map-empty-state">
                <span class="map-empty-icon">🏠</span>
                <b>우리 집 맵을 준비하고 있어요</b>
                <span>맞춤 청소 준비에서 1회차 학습을 시작하면<br>방 구조와 바닥 상태가 여기에 표시돼요.</span>
              </div>
            </div>
          </section>

          <section class="panel plan-panel prep-section" aria-labelledby="prepSectionTitle">
            <div class="plan-head">
              <div>
                <div class="home-section-kicker">AI HOME PROFILE</div>
                <div class="plan-title" id="prepSectionTitle">✨ 우리 집 맞춤청소 준비</div>
              </div>
              <div class="plan-model" id="planModel">1회차 학습 전</div>
            </div>
            <div class="learn-panel" id="learnPanel">
              <div class="learn-top">
                <div class="learn-title" id="learnTitle">처음 사용할 때는 로보킹이 집을 먼저 배워요</div>
                <div class="learn-pill" id="learnPill">초기 학습</div>
              </div>
              <div class="learn-desc" id="learnDesc">처음 한 번만 집 구조와 바닥 상태를 배워요.</div>
              <div class="learn-progress"><div class="learn-fill" id="learnFill"></div></div>
              <div class="learn-status" id="learnStatus">1회차 학습 청소를 시작하면 로보킹이 집 구조와 구역 정보를 자동으로 기록해요.</div>
              <div class="learn-steps" id="learnSteps"></div>
              <div class="learn-actions" id="learnActions">
                <button type="button" class="learn-btn" id="learnBtn" data-action="startFirstMapping" onpointerdown="window.__forceStartFirstMapping && window.__forceStartFirstMapping(event);" onmousedown="window.__forceStartFirstMapping && window.__forceStartFirstMapping(event);" ontouchstart="window.__forceStartFirstMapping && window.__forceStartFirstMapping(event);" onclick="window.__forceStartFirstMapping && window.__forceStartFirstMapping(event);">🏠 1회차 학습 청소 시작</button>
                <button type="button" class="clean-execute-btn" id="cleanExecuteBtn" data-action="executeTopClean" style="display:none;">🧹 청소하기</button>
              </div>
            </div>
            <div class="scope-buttons">
              <button class="scope-btn active" id="scopeHome" data-action="selectHome">집 전체</button>
              <button class="scope-btn" id="scopeZone1" data-action="selectZone1">1구역</button>
              <button class="scope-btn" id="scopeZone2" data-action="selectZone2">2구역</button>
              <button class="scope-btn" id="scopeZone3" data-action="selectZone3">3구역</button>
              <button class="scope-btn" id="scopeZone4" data-action="selectZone4">4구역</button>
              <button class="scope-btn" id="scopeZone5" data-action="selectZone5">5구역</button>
            </div>
            <div class="selected-plan">
              <div class="plan-summary" id="planSummary">집 전체 청소 조건을 분석 중입니다.</div>
              <div class="plan-soc">
                <div class="plan-soc-label">충전 준비</div>
                <div class="plan-soc-value"><span id="planTargetSoc">81</span>%</div>
                <div class="plan-soc-sub" id="planSocSub">필요한 만큼만 충전</div>
              </div>
            </div>
            <button class="start-clean-primary" id="startCleanPrimary" data-action="clean" disabled>
              🧹 청소 미션 수행하기
              <small id="startCleanHint">준비가 끝나면 바로 시작할 수 있어요</small>
            </button>
          </section>

          <section class="panel condition-panel direct-clean-section" id="conditionPanel" aria-labelledby="conditionTitle">
              <div class="home-section-kicker">MANUAL CLEANING</div>
              <div class="condition-title" id="conditionTitle">✍️ 직접 조건 청소</div>

              <div id="firstLearnInputs">
                <div class="condition-help first-learn-note compact-note">
                  <div class="note-title">맞춤 청소 준비 후 사용할 수 있어요</div>
                  <div class="note-caption">먼저 위의 1회차 학습을 완료하면 청소 범위·방식·강도·오늘 상태를 직접 선택할 수 있어요.</div>
                </div>
              </div>

              <div id="predictionInputs" style="display:none;">
                <div class="condition-help">세부 조건을 직접 고르면 로보킹이 준비부터 청소까지 이어서 진행해요.</div>
                <div class="predict-condition-grid">
                  <label for="scopeSelect">청소 범위</label>
                  <select class="condition-select" id="scopeSelect">
                    <option value="home">집 전체</option>
                    <option value="1">1구역</option>
                    <option value="2">2구역</option>
                    <option value="3">3구역</option>
                    <option value="4">4구역</option>
                    <option value="5">5구역</option>
                  </select>
                  <label for="cleanModeSelect">청소 방식</label>
                  <select class="condition-select" id="cleanModeSelect">
                    <option value="dry">건식</option>
                    <option value="mop">물걸레</option>
                    <option value="both">건식+물걸레</option>
                  </select>
                  <label for="intensitySelect">청소 강도</label>
                  <select class="condition-select" id="intensitySelect">
                    <option value="fast">빠른</option>
                    <option value="standard" selected>표준</option>
                    <option value="careful">꼼꼼</option>
                  </select>
                  <label for="todayStateSelect">오늘 상태</label>
                  <select class="condition-select" id="todayStateSelect">
                    <option value="normal">평소와 같음</option>
                    <option value="dust">먼지 많음</option>
                    <option value="pet">반려동물 털 많음</option>
                    <option value="obstacle">바닥 물건 많음</option>
                  </select>
                </div>
              </div>

              <button class="predict-btn manual-combo-btn" id="predictBtn" data-action="manualCleanAndGo">🔥 선택 조건으로 준비하고 청소하기</button>
              <div class="predict-loading" id="predictLoading">1회차 학습 청소가 끝나면 오늘 청소 준비를 할 수 있어요.</div>
              <div class="flow-guide" id="flowGuide"><span class="guide-step">현재 단계</span>1회차 학습 청소로 집 정보를 먼저 저장해 주세요.</div>
          </section>

          <div class="actions">
            <button class="action-btn" data-action="feed"><span class="action-icon">🥣</span>먹여주기</button>
            <button class="action-btn" data-action="play"><span class="action-icon">🏐</span>놀아주기</button>
            <button class="action-btn" data-action="train"><span class="action-icon">🏋️</span>훈련하기</button>
            <button class="action-btn" data-action="photo"><span class="action-icon">📷</span>사진첩</button>
            <button class="action-btn" data-action="clean"><span class="action-icon">🏆</span>미션</button>
            <button class="action-btn" data-action="shop"><span class="action-icon">🛒</span>상점</button>
          </div>

          <div class="home-cards">
            <section class="mini-card">
              <div class="mini-title">배터리 컨디션</div>
              <div class="battery-info">너무 배부르거나<br>너무 배고프지 않게<br>로보킹이 알아서 관리해요!</div>
              <div class="battery-face" id="batteryFace">😊</div>
              <div class="scale"><div class="pointer" id="pointer"></div></div>
              <div class="scale-labels"><span>0%</span><span>15%</span><span>90%</span><span>100%</span></div>
              <div class="battery-message" id="batteryMessage">배터리 컨디션이 좋아요.</div>
            </section>

            <section class="mini-card time-card">
              <div class="mini-title">필요 청소 가능 시간</div>
              <div class="time-icon">🤖</div>
              <div class="time-number"><span id="cleanTime">45</span><small> 분</small></div>
              <div class="time-sub">현재 배터리 기준</div>
              <div class="time-tip" id="timeTip">현재 배터리로 청소가 가능합니다.</div>
            </section>

            <section class="mini-card food-card">
              <div class="food-title">오늘의 음식</div><div class="food-bowl"></div>
              <div class="food-count">보유량: <span id="foodText">1</span>개</div>
            </section>
          </div>
        </div>
      </section>

      <!-- ===================== PAGE 2 · 부품 케어 ===================== -->
      <section class="page" id="batteryPage">
        <div class="section-kicker">PARTS CARE</div>
        <div class="section-title">부품 상태 확인</div>

        <div class="parts-grid" id="partsGrid"></div>

        <div class="panel care-summary">
          <div class="panel-head"><div class="panel-title">배터리 수명 지키기</div><div class="badge">과충전 방지</div></div>
          <div class="care-lead">완충(100%) 대신 청소에 필요한 만큼만 채우고, 15%를 남기고 쉬어가요. 이렇게 배터리 수명을 늘리고 있어요.</div>
          <div class="care-stats">
            <div class="care-stat"><span>맞춤 충전</span><b id="careAcceptText">4</b><small>회</small></div>
            <div class="care-stat"><span>잔량 15% 보호</span><b id="careReserveText">1</b><small>회</small></div>
            <div class="care-stat"><span>덜 채운 충전량</span><b id="careSavedText">76</b><small>%</small></div>
          </div>
          <div class="care-health-row">
            <span>배터리 건강도</span>
            <div class="care-health-track"><div class="care-health-fill" id="careHealthFill" style="width:100%"></div></div>
            <b id="careHealthText">100%</b>
          </div>
          <div class="care-note" id="careNote">오늘도 과충전 없이 관리 중이에요.</div>
        </div>

        <div class="panel events">
          <div class="panel-head"><div class="panel-title">실시간 케어 기록</div><div class="badge">자동 기록</div></div>
          <div id="eventList">
            <div class="event-item"><div class="event-time">14:20</div><div class="event-content"><strong>맞춤 충전 완료<span class="event-tag">수명 보호</span></strong><span>81%까지만 채우고 멈췄어요. 완충 대비 19% 덜 채워 과충전을 막았어요.</span></div></div>
            <div class="event-item"><div class="event-time">10:15</div><div class="event-content"><strong>청소 준비 완료<span class="event-tag">배터리 절약</span></strong><span>거실 상태에 맞춰 필요한 배터리만 계산했어요.</span></div></div>
            <div class="event-item"><div class="event-time">08:40</div><div class="event-content"><strong>배터리 컨디션 정상<span class="event-tag">온도 안정</span></strong><span>배터리 온도 29℃, 안정 범위(15~50℃) 안에 있어요.</span></div></div>
          </div>
        </div>
      </section>

      <!-- ===================== PAGE 3 · 예약 청소 ===================== -->
      <section class="page" id="recordPage">
        <div class="section-kicker">SMART SCHEDULE</div>
        <div class="section-title">출퇴근 맞춤 예약 청소</div>

        <div class="panel sched-panel">
          <div class="sched-head">
            <div>
              <div class="sched-title">🚶 출퇴근 맞춤 예약</div>
              <div class="sched-desc">집을 비우는 시간에만 청소하고, 돌아오기 전에 조용히 도킹해 두어요.</div>
            </div>
            <button class="switch" id="commuteSwitch" data-action="toggleCommute" aria-label="출퇴근 예약 켜기"><span class="knob"></span></button>
          </div>
          <div class="sched-body off" id="commuteBody">
            <div class="time-row">
              <label for="leaveTime">출근</label>
              <select class="condition-select" id="leaveTime"></select>
              <label for="returnTime">퇴근</label>
              <select class="condition-select" id="returnTime"></select>
            </div>
            <div class="day-chips" id="dayChips"></div>
            <div class="sched-options">
              <button class="sched-opt active" id="commuteAfter" data-action="commuteMode" data-mode="after">출근 30분 뒤 시작</button>
              <button class="sched-opt" id="commuteBefore" data-action="commuteMode" data-mode="before">퇴근 1시간 전 마무리</button>
            </div>
          </div>
          <div class="sched-preview off" id="commutePreview">스위치를 켜면 출퇴근 시간에 맞춘 예약이 만들어져요.</div>
        </div>

        <div class="sub-title">🎎 테마별 기간 청소</div>
        <div class="theme-list" id="themeList"></div>

        <div class="panel upcoming">
          <div class="panel-head"><div class="panel-title">다가오는 예약</div><div class="badge" id="upcomingBadge">0건</div></div>
          <div id="upcomingList"></div>
        </div>
      </section>

      <!-- ===================== PAGE 4 · 이벤트 ===================== -->
      <section class="page" id="eventPage">
        <div class="section-kicker">DISCOVERY & MISSION</div>
        <div class="section-title">이벤트</div>

        <div class="event-tabs">
          <button class="reward-folder-btn active" id="evTabFound" data-action="evTabFound">🔍 오늘의 발견</button>
          <button class="reward-folder-btn" id="evTabMission" data-action="evTabMission">🏅 미션</button>
          <button class="reward-folder-btn" id="evTabPhoto" data-action="evTabPhoto">📷 사진첩</button>
        </div>

        <div class="reward-panel" id="evFoundPanel">
          <div class="found-top">
            <div class="found-card" id="foundTodayCard"></div>
            <div class="found-card">
              <div class="found-card-title">📍 발견 위치 보기</div>
              <div class="found-map" id="foundMap"></div>
              <div class="found-btn"><button data-action="foundMapBig">지도 크게 보기 ›</button></div>
            </div>
          </div>
          <div class="panel found-list">
            <div class="panel-head"><div class="panel-title">📋 최근 발견 기록</div><div class="badge" id="foundCountBadge">전체 3건</div></div>
            <div id="foundList"></div>
          </div>
        </div>

        <div class="reward-panel hidden" id="evMissionPanel">
          <div class="panel mission-summary">
            <div>
              <div class="ms-title">도전과제 메달</div>
              <div class="ms-desc">목표를 달성하면 메달과 코인을 받아요.<br>받은 코인은 리워드에서 쓸 수 있어요.</div>
            </div>
            <div class="ms-count"><b id="missionClaimable">0</b>받을 보상</div>
          </div>
          <div id="missionList"></div>
        </div>

        <div class="reward-panel hidden" id="evPhotoPanel">
          <div class="panel photo-info">
            <div>
              <div class="p-title">📷 로보킹 사진첩</div>
              <div class="p-desc">청소 중 움직이는 친구를 만나면 로보킹이 살짝 찍어둬요. 혼자 있는 반려동물의 하루를 볼 수 있어요.</div>
            </div>
            <div class="p-count"><b id="photoCount">0</b>장</div>
          </div>
          <div class="photo-grid" id="photoGrid"></div>
          <div class="panel photo-empty" id="photoEmpty" style="display:none;">
            지금은 예시 사진이에요.<br>실제 사진을 넣으려면 앱 폴더의 <code>assets/photos/</code> 안에 사진 파일을 넣어주세요.
          </div>
        </div>
      </section>

      <!-- ===================== PAGE 5 · 리워드 (원본 유지) ===================== -->
      <section class="page" id="rewardPage">
        <div class="section-kicker">REWARD</div>
        <div class="section-title">로보킹 성장 리워드</div>

        <div class="panel level-panel">
          <div class="level-robot-preview" id="levelRobotPreview"><span class="preview-base">🤖</span></div>
          <div class="level-number">Lv. <span id="levelText">13</span></div>
          <div class="level-track"><div class="level-fill" id="expFill"></div></div>
          <div class="level-caption">경험치 <span id="expText">55</span> / 100</div>
        </div>

        <div class="reward-folder-tabs">
          <button class="reward-folder-btn active" id="rewardTabItems" data-action="rewardTabItems">꾸미기 아이템</button>
          <button class="reward-folder-btn" id="rewardTabCoupons" data-action="rewardTabCoupons">LG 혜택 쿠폰</button>
        </div>

        <div class="reward-panel" id="rewardItemsPanel">
          <div class="reward-grid">
            <div class="reward-card" id="cardFood"><div class="reward-icon">🥣</div><div class="reward-title">에너지 간식</div><div class="reward-desc">먹으면 배터리가 조금 회복돼요.</div><div class="reward-status" id="statusFood"></div><button class="reward-btn" id="btnFood" data-action="buyFood">50 코인</button></div>
            <div class="reward-card" id="cardRibbon"><div class="reward-icon">🎀</div><div class="reward-title">빨간 리본</div><div class="reward-desc">머리 위에 귀엽게 달아줘요.</div><div class="reward-status" id="statusRibbon"></div><button class="reward-btn" id="btnRibbon" data-action="itemRibbon">60 코인</button></div>
            <div class="reward-card" id="cardHat"><div class="reward-icon">🧢</div><div class="reward-title">탐험가 모자</div><div class="reward-desc">로보킹 머리에 딱 맞게 씌워줘요.</div><div class="reward-status" id="statusHat"></div><button class="reward-btn" id="btnHat" data-action="itemHat">120 코인</button></div>
            <div class="reward-card" id="cardSparkle"><div class="reward-icon">✨</div><div class="reward-title">반짝이 오라</div><div class="reward-desc">로보킹 주변이 반짝여요.</div><div class="reward-status" id="statusSparkle"></div><button class="reward-btn" id="btnSparkle" data-action="itemSparkle">80 코인</button></div>
            <div class="reward-card" id="cardBunny"><div class="reward-icon">🐰</div><div class="reward-title">토끼 귀</div><div class="reward-desc">로보킹 머리에 토끼 귀가 쏙!</div><div class="reward-status" id="statusBunny"></div><button class="reward-btn" id="btnBunny" data-action="itemBunny">90 코인</button></div>
            <div class="reward-card" id="cardCat"><div class="reward-icon">🐱</div><div class="reward-title">고양이 귀</div><div class="reward-desc">새침한 고양이 로보킹으로 변신!</div><div class="reward-status" id="statusCat"></div><button class="reward-btn" id="btnCat" data-action="itemCat">70 코인</button></div>
          </div>
        </div>

        <div class="reward-panel hidden" id="rewardCouponsPanel">
          <div class="reward-grid">
            <div class="reward-card coupon-card" id="cardCouponLg5"><div class="reward-icon">🎟</div><div class="reward-title">LG 생활가전 5% 쿠폰</div><div class="reward-desc">LG 생활가전 1개를 구매할 때 사용할 수 있는 기본 할인 쿠폰이에요.</div><div class="coupon-benefit">혜택: 단일 제품 5% 할인</div><div class="reward-status" id="statusCouponLg5"></div><button class="reward-btn" id="btnCouponLg5" data-action="couponLg5">300 코인</button></div>
            <div class="reward-card coupon-card" id="cardCouponCleanKit"><div class="reward-icon">🧹</div><div class="reward-title">로보킹 클린 키트 쿠폰</div><div class="reward-desc">필터, 브러시, 물걸레 패드처럼 자주 바꾸는 소모품을 준비할 때 사용해요.</div><div class="coupon-benefit">혜택: 소모품 키트 구매 할인</div><div class="reward-status" id="statusCouponCleanKit"></div><button class="reward-btn" id="btnCouponCleanKit" data-action="couponCleanKit">180 코인</button></div>
            <div class="reward-card coupon-card" id="cardCouponBatteryCare"><div class="reward-icon">🔋</div><div class="reward-title">배터리 케어 쿠폰</div><div class="reward-desc">로보킹을 오래 쓰기 위해 배터리 점검이나 관리 서비스를 받을 때 사용해요.</div><div class="coupon-benefit">혜택: 배터리 점검/케어 서비스</div><div class="reward-status" id="statusCouponBatteryCare"></div><button class="reward-btn" id="btnCouponBatteryCare" data-action="couponBatteryCare">250 코인</button></div>
            <div class="reward-card coupon-card" id="cardCouponMoveIn"><div class="reward-icon">🏠</div><div class="reward-title">이사·입주 패키지 쿠폰</div><div class="reward-desc">새집에 필요한 LG 생활가전을 2개 이상 함께 구매할 때 추가 혜택을 받아요.</div><div class="coupon-benefit">혜택: 2개 이상 구매 시 패키지 추가 할인</div><div class="reward-status" id="statusCouponMoveIn"></div><button class="reward-btn" id="btnCouponMoveIn" data-action="couponMoveIn">300 코인</button></div>
          </div>
        </div>
      </section>

    </main>

    <div class="modal" id="modal">
      <div class="modal-card">
        <div class="modal-title" id="modalTitle"></div>
        <div class="modal-body" id="modalBody"></div>
        <div class="modal-actions single" id="modalActions">
          <button class="modal-btn modal-secondary" id="modalCancel">취소</button>
          <button class="modal-btn modal-primary" id="modalConfirm">확인</button>
        </div>
      </div>
    </div>
    <div class="toast" id="toast"></div>
  </div>
</div>

<script>
"use strict";

const predictionData = __UI_PREDICTION_DATA__;
const mediaData = __UI_MEDIA_DATA__;
let activeRun = null;
const mappingSteps=[
  {key:'map',label:'집 구조 매핑'},
  {key:'area',label:'구역별 면적 저장'},
  {key:'floor',label:'바닥 타입 인식'},
  {key:'dirt',label:'오염도 기록'},
  {key:'obstacle',label:'장애물 수준 기록'},
  {key:'soc',label:'배터리 사용 기록'}
];

const $=(id)=>document.getElementById(id);
const clamp=(v,min,max)=>Math.min(Math.max(v,min),max);
const fmtSoc=(v)=>Number(v || 0).toFixed(1).replace(/\.0$/,"");
const cleanMinutes=()=>Math.max(0,Math.round(state.soc*.56));
const setHtml=(el,html)=>{if(el&&el.__lastHtml!==html){el.innerHTML=html;el.__lastHtml=html;}};
const esc=(s)=>String(s==null?"":s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#39;");

// ============================================================
// 배터리 보호/학습 주행 기준값
// 최근 UX 문구 수정 과정에서 이 상수들이 빠지면
// 1회차 학습 버튼 클릭 시 startFirstMapping() 내부에서 ReferenceError가 발생합니다.
// 그래서 사용자에게 숫자를 직접 노출하지 않더라도, 내부 로직에는 반드시 유지합니다.
// ============================================================
const MIN_RESERVE_SOC = 15;
const MAX_CHARGE_SOC = 90;
const MAX_SINGLE_PASS_USE = MAX_CHARGE_SOC - MIN_RESERVE_SOC;
const CRITICAL_DOCK_SOC = MIN_RESERVE_SOC;
const targetFromRequired = (required)=>clamp(Math.ceil(Number(required||0)+MIN_RESERVE_SOC),MIN_RESERVE_SOC,MAX_CHARGE_SOC);
const expectedEndSoc = (startSoc,required)=>Math.round((Number(startSoc||0)-Number(required||0))*10)/10;

const MIN_SOC_AFTER_LEARNING = MIN_RESERVE_SOC;
const MIN_LEARNING_SOC_USE = 5;
const MAX_LEARNING_SOC_USE = 30;
const LEARNING_SOC_RATIO = 0.35;

// 시연용: 이전에 이미 몇 번 청소한 로봇처럼 보이게 하는 누적 청소 횟수 기준값
// (미션 "10번 청소하기"가 시연 중 첫 청소로 달성되도록 9로 둡니다)
const DEMO_CLEAN_BASE = 9;

function setModeChipText(text){
  const el=$("modeChip");
  if(el)el.textContent=text;
}

function setGuide(message,tone="normal"){
  state.userGuide=message;
  state.userGuideTone=tone;
  const guide=$('flowGuide');
  if(guide){
    guide.className="flow-guide"+(tone&&tone!=="normal"?" "+tone:"");
    guide.innerHTML="<span class='guide-step'>다음 안내</span>"+message;
  }
}
function guideForCurrentState(){
  if(state.mapping)return "로보킹이 우리 집을 배우는 중이에요. 집 구조와 바닥 상태를 차근차근 기억하고 있어요.";
  if(!state.profileReady)return "<b>1단계</b> 먼저 1회차 학습 청소로 우리 집을 알려주세요.";
  if(state.profileReady && !state.predicted)return "<b>2단계</b> AI 자동청소를 고르거나, 바로 청소하기를 눌러주세요.";
  if(state.charging)return "로보킹이 잠깐 쉬면서 힘을 채우고 있어요. 필요한 만큼 채우면 알아서 멈춰요.";
  if(state.cleaning)return "청소 중이에요. 배터리가 무리하지 않도록 로보킹이 알아서 조절하고 있어요.";
  if(state.celebrating || state.missionDone)return "청소가 끝났어요! 로보킹이 배터리를 아끼며 마무리했어요.";
  if(state.predicted && state.soc<state.targetSoc)return "<b>3단계</b> 준비가 끝났어요. 청소하기를 누르면 필요한 만큼만 채우고 출발해요.";
  if(state.predicted)return "<b>3단계</b> 지금 바로 출동할 수 있어요. 청소하기를 눌러주세요.";
  return state.userGuide||"현재 상태를 확인 중입니다.";
}

function getLearningSocUse(run,startSoc){
  const fullRequired=Math.max(0,Number(run&&run.home?run.home.requiredSoc:0));
  const available=Math.max(0,Number(startSoc||0)-MIN_SOC_AFTER_LEARNING);
  if(fullRequired<=0 || available<=0)return 0;
  let mappedUse=fullRequired*LEARNING_SOC_RATIO;
  mappedUse=Math.max(MIN_LEARNING_SOC_USE,mappedUse);
  mappedUse=Math.min(mappedUse,MAX_LEARNING_SOC_USE,fullRequired,available);
  return Math.round(mappedUse*10)/10;
}

const cleanModeLabels={dry:"건식",mop:"물걸레",both:"건식+물걸레"};
const intensityLabels={fast:"빠른",standard:"표준",careful:"꼼꼼"};
const todayStateLabels={normal:"평소와 같음",dust:"먼지 많음",pet:"반려동물 털 많음",obstacle:"바닥 물건 많음"};
// 아래 선택값은 배율을 곱하는 보정계수가 아니라,
// 기록 안에서 조건이 가장 가까운 우리 집 기록 준비 행을 찾기 위한 검색 조건으로 사용됩니다.
const intensityAliases={fast:["약","중"],standard:["중","강"],careful:["강","터보"]};
const todayStateAliases={normal:"학습 프로필 기준",dust:"오염도 높은 조건",pet:"오염도 높음 + 강한 흡입 조건",obstacle:"장애물 많은 조건"};

const closetDefault={
  owned:{ribbon:false,hat:false,bunny:false,cat:false,sparkle:false},
  equipped:{head:"crown",aura:null}
};
const shopItems={
  ribbon:{name:"빨간 리본",icon:"🎀",cost:60,slot:"head",value:"ribbon",message:"빨간 리본을 달아줬어요! 로보킹이 더 사랑스러워졌어요."},
  hat:{name:"탐험가 모자",icon:"🧢",cost:120,slot:"head",value:"hat",message:"탐험가 모자를 씌워줬어요! 이제 진짜 모험가 로보킹이에요."},
  bunny:{name:"토끼 귀",icon:"🐰",cost:90,slot:"head",value:"bunny",message:"토끼 귀를 달아줬어요! 로보킹이 통통 튀는 기분이에요."},
  cat:{name:"고양이 귀",icon:"🐱",cost:70,slot:"head",value:"cat",message:"고양이 귀를 달아줬어요! 로보킹이 더 새침해졌어요."},
  sparkle:{name:"반짝이 오라",icon:"✨",cost:80,slot:"aura",value:"sparkle",message:"반짝이 오라를 켰어요! 청소할 때마다 기분이 좋아져요."}
};

const couponItems={
  lg5:{name:"LG 생활가전 5% 쿠폰",icon:"🎟",cost:300,benefit:"LG 생활가전 1개 구매 시 5% 할인",message:"LG 생활가전 5% 쿠폰을 보관함에 담았어요."},
  cleanKit:{name:"로보킹 클린 키트 쿠폰",icon:"🧹",cost:180,benefit:"필터·브러시·물걸레 패드 등 소모품 키트 할인",message:"로보킹 클린 키트 쿠폰을 보관함에 담았어요."},
  batteryCare:{name:"배터리 케어 쿠폰",icon:"🔋",cost:250,benefit:"배터리 점검 또는 관리 서비스 혜택",message:"배터리 케어 쿠폰을 보관함에 담았어요."},
  moveIn:{name:"이사·입주 패키지 쿠폰",icon:"🏠",cost:300,benefit:"LG 생활가전 2개 이상 구매 시 패키지 추가 혜택",message:"이사·입주 패키지 쿠폰을 보관함에 담았어요."}
};
function loadCoupons(){
  try{
    const raw=localStorage.getItem("lgRoboCareCouponsV1");
    const defaults={lg5:0,cleanKit:0,batteryCare:0,moveIn:0};
    if(!raw)return defaults;
    return Object.assign(defaults,JSON.parse(raw)||{});
  }catch(e){return {lg5:0,cleanKit:0,batteryCare:0,moveIn:0};}
}
function saveCoupons(){
  try{localStorage.setItem("lgRoboCareCouponsV1",JSON.stringify(state.ownedCoupons));}catch(e){}
}

function loadCloset(){
  try{
    const raw=localStorage.getItem("lgRoboCareClosetV2");
    if(!raw)return JSON.parse(JSON.stringify(closetDefault));
    const saved=JSON.parse(raw);
    const owned=Object.assign({},closetDefault.owned,saved.owned||{});
    // 이전 버전에서 하트 스티커를 샀다면 고양이 귀 보유로 자연스럽게 이전합니다.
    if(saved.owned && saved.owned.heart && !owned.cat)owned.cat=true;
    const equipped=Object.assign({},closetDefault.equipped,saved.equipped||{});
    if(equipped.decal==="heart" && (!equipped.head || equipped.head==="crown"))equipped.head="cat";
    delete equipped.decal;
    return {owned,equipped};
  }catch(e){return JSON.parse(JSON.stringify(closetDefault));}
}
function saveCloset(){
  try{localStorage.setItem("lgRoboCareClosetV2",JSON.stringify({owned:state.ownedItems,equipped:state.equippedItems}));}catch(e){}
}
const initialCloset=loadCloset();
const initialCoupons=loadCoupons();

function pickRandomRun(candidates){
  if(!candidates || candidates.length===0)return null;

  // 현재 화면에 떠 있는 global_run_id와 같은 시나리오는 가능하면 제외
  // 같은 조건으로 1회차 학습 청소를 다시 실행할 때 매번 다른 집 정보가 나오게 하기 위함
  let pool = candidates;
  if(activeRun && activeRun.globalRunId && candidates.length>1){
    const filtered = candidates.filter(r=>String(r.globalRunId)!==String(activeRun.globalRunId));
    if(filtered.length>0)pool = filtered;
  }

  const idx = Math.floor(Math.random()*pool.length);
  return pool[idx];
}

function findRun(areaPyung, mopEnabled){
  const area = Number(areaPyung);
  const mop = Boolean(mopEnabled);

  // 1순위: 사용자가 선택한 평수 + 청소방식이 모두 같은 기록 시나리오 중 랜덤 선택
  let candidates = predictionData.runs.filter(r=>Number(r.areaPyung)===area && Boolean(r.mopEnabled)===mop);
  let run = pickRandomRun(candidates);

  // 2순위: 청소방식까지 완전히 맞는 데이터가 없으면, 같은 평수 안에서 랜덤 선택
  if(!run){
    candidates = predictionData.runs.filter(r=>Number(r.areaPyung)===area);
    run = pickRandomRun(candidates);
  }

  // 3순위: 같은 평수도 없으면 전체 기록 중 랜덤 선택
  if(!run){
    run = pickRandomRun(predictionData.runs);
  }

  return run || predictionData.runs[0];
}

activeRun = pickRandomRun(predictionData.runs) || predictionData.runs[0];

const state={
  page:"homePage",
  soc:predictionData.currentSoc,
  targetSoc:activeRun.home.targetSoc,
  requiredSoc:activeRun.home.requiredSoc,
  selectedScope:"home",
  selectedZone:null,
  selectedLabel:activeRun.home.label,
  selectedScenario:activeRun.home,
  modelName:activeRun.home.modelName,
  globalRunId:activeRun.home.globalRunId,
  areaPyung:activeRun.home.areaPyung,
  cleaningAreaM2:activeRun.home.cleaningAreaM2,
  cleaningType:activeRun.home.cleaningType,
  mopEnabled:activeRun.home.mopEnabled,
  obstacleLevel:activeRun.home.obstacleLevel,
  floorType:activeRun.home.floorType,
  dirtLevel:activeRun.home.dirtLevel,
  suctionMode:activeRun.home.suctionMode,
  cleanModeChoice:activeRun.home.mopEnabled?'mop':'dry',
  cleanModeLabel:activeRun.home.cleaningType,
  intensityChoice:'standard',
  intensityLabel:'표준',
  todayStateChoice:'normal',
  todayStateLabel:'평소와 같음',
  matchNote:"우리 집 기록으로 맞춤 준비",
  matchBasis:"오늘 상태 반영",
  cleaningRemainingSoc:activeRun.home.requiredSoc,
  cleaningSegmentIndex:0,
  splitCleaning:false,
  pendingCleanAfterCharge:false,
  chargeComplete:false,
  predicting:false,
  predicted:false,
  userGuide:"1회차 학습 청소로 집 크기, 구역, 바닥, 오염도, 배터리 사용을 저장해 주세요.",
  userGuideTone:"normal",
  profileReady:false,
  mapping:false,
  mappingProgress:0,
  mappingStepIndex:-1,
  firstRunSocUsed:0,
  firstRunRequiredSoc:0,
  firstRunFullRequiredSoc:0,
  firstRunStartSoc:predictionData.currentSoc,
  firstRunEndSoc:predictionData.currentSoc,
  firstRunSocEnough:true,
  ownedItems:initialCloset.owned,
  equippedItems:initialCloset.equipped,
  rewardTab:"items",
  ownedCoupons:initialCoupons,
  noGoZones:[],
  mapMode:"view",
  smartCleanMode:"auto",
  selectedDirtyZones:[],
  manualReady:false,
  manualKey:"",
  robotMotion:"idle",
  cleaningZones:[],
  currentCleaningZone:null,
  completedZones:[],
  cleanAnim:null,
  chargePurpose:"current",
  nextHomeReady:false,
  nextHomeTargetSoc:0,
  nextHomeRequiredSoc:0,
  temperature:29,health:100,heart:100,
  level:13,exp:55,coins:50,food:1,cleaning:false,charging:false,
  celebrating:false,progress:0,missionDone:false,cleanCount:0,
  acceptCount:4,area:activeRun.home.cleaningAreaM2||72,average:38,

  // ---- 새 페이지(부품 케어 / 예약 청소 / 이벤트) 전용 상태 ----
  learnCount:0,
  savedChargePct:76,
  reserveGuardCount:1,
  claimedMissions:{},
  notifiedClaimable:0,
  eventTab:"found",
  commuteOn:false,
  commuteDays:[1,2,3,4,5],
  commuteMode:"after",
  activeThemes:{}
};

function populateConditionSelectors(){
  refreshScopeSelect();
  const cleanModeSelect=$('cleanModeSelect');
  if(cleanModeSelect)cleanModeSelect.value=activeRun.mopEnabled?'mop':'dry';
  const scopeSelect=$('scopeSelect');
  if(scopeSelect)scopeSelect.value='home';
  const intensitySelect=$('intensitySelect');
  if(intensitySelect)intensitySelect.value='standard';
  const todayStateSelect=$('todayStateSelect');
  if(todayStateSelect)todayStateSelect.value='normal';
}

function switchPage(pageId){
  if(!$(pageId))return;
  document.querySelectorAll(".page").forEach(p=>p.classList.remove("active"));
  document.querySelectorAll(".nav-btn").forEach(b=>b.classList.toggle("active",b.dataset.page===pageId));
  $(pageId).classList.add("active");
  state.page=pageId;
  render();
}

function render(){
  state.soc=clamp(Math.round(state.soc),0,100);
  state.targetSoc=clamp(Math.round(state.targetSoc),15,90);
  state.temperature=Math.round(state.temperature*10)/10;
  state.exp=clamp(Math.round(state.exp),0,100);

  $("coinText").textContent=String(state.coins).padStart(3,"0");
  $("foodText").textContent=state.food;
  $("cleanTime").textContent=cleanMinutes();
  $("pointer").style.left=state.soc+"%";
  $("cleanFill").style.width=state.progress+"%";
  $("missionFill").style.width=state.missionDone?"100%":state.progress+"%";
  const robotSocBadge=$("robotSocBadge");
  if(robotSocBadge){
    const socState=state.soc<15?"low":(state.soc<state.targetSoc?"need":"ok");
    robotSocBadge.className="robot-soc-badge "+socState;
    robotSocBadge.innerHTML="<span>🔋 현재 배터리</span><b>"+state.soc+"%</b>";
  }

  renderAccessories();renderPlan();renderHome();
  renderCare();renderSchedule();renderEvents();
  renderReward();
}

function getScenario(scope,zoneNumber=null){
  if(scope==="home")return activeRun.home;
  const zones=(activeRun && activeRun.zones) ? activeRun.zones : [];
  const direct=zones.find(z=>Number(z.zone)===Number(zoneNumber)) || zones[zoneNumber-1];
  if(direct)return direct;

  // 혹시 예전 CSV/캐시로 해당 영역 데이터가 아직 없으면 가장 가까운 영역 데이터를 임시로 사용합니다.
  // 새 4/6/8구역 CSV로 교체되면 실제 해당 영역 데이터가 자동으로 잡힙니다.
  if(zones.length){
    const idx=Math.min(Math.max(Number(zoneNumber||1)-1,0),zones.length-1);
    return zones[idx];
  }
  return activeRun.home;
}

function getPredictionChoices(scopeOverride=null,zoneOverride=null){
  const scopeSelect=$('scopeSelect');
  const cleanModeSelect=$('cleanModeSelect');
  const intensitySelect=$('intensitySelect');
  const todayStateSelect=$('todayStateSelect');
  const scopeValue=scopeOverride? (scopeOverride==="home"?"home":String(zoneOverride||1)) : (scopeSelect?scopeSelect.value:"home");
  return {
    scopeValue,
    scope:scopeValue==="home"?"home":"zone",
    zoneNumber:scopeValue==="home"?null:Number(scopeValue),
    cleanMode:cleanModeSelect?cleanModeSelect.value:(activeRun.home.mopEnabled?'mop':'dry'),
    intensity:intensitySelect?intensitySelect.value:'standard',
    todayState:todayStateSelect?todayStateSelect.value:'normal'
  };
}

function getBaseScenarioFromChoices(choices){
  if(choices.scope==="home")return activeRun.home;
  return getScenario("zone",choices.zoneNumber);
}

function getCleanModeCandidateValue(scenario){
  const txt=String(scenario.cleaningType||"");
  if(txt.includes("건식+물걸레")||txt.includes("복합")||txt.includes("both"))return "both";
  if(scenario.mopEnabled || txt.includes("물걸레") || txt.toLowerCase().includes("mop"))return "mop";
  return "dry";
}

function cleanModeScore(scenario,choice){
  const v=getCleanModeCandidateValue(scenario);
  if(choice===v)return 120;
  // 데이터셋에 건식+물걸레가 없을 수 있으므로, 복합 청소는 물걸레 조건을 가장 가까운 후보로 인정합니다.
  if(choice==="both" && v==="mop")return 70;
  return -260;
}

function suctionPreferenceScore(scenario,intensity){
  const mode=String(scenario.suctionMode||"");
  const code=Number(scenario.suctionCode||scenario.suctionMaxCode||0);
  if(intensity==="fast"){
    if(mode.includes("약"))return 80;
    if(mode.includes("중"))return 50;
    if(code && code<=2)return 60;
    return -25;
  }
  if(intensity==="careful"){
    if(mode.includes("터보"))return 90;
    if(mode.includes("강"))return 65;
    if(code && code>=3)return 70;
    return -20;
  }
  // 표준은 중/강 또는 학습 프로필과 가까운 후보를 우선합니다.
  if(mode.includes("중"))return 70;
  if(mode.includes("강"))return 45;
  if(code && code>=2 && code<=3)return 55;
  return 10;
}

function todayStateScore(scenario,choice,baseScenario){
  const dirt=Number(scenario.dirtCode||scenario.dirtMaxCode||0);
  const dirtMax=Number(scenario.dirtMaxCode||dirt||0);
  const obs=Number(scenario.obstacleLevelCode||0);
  const suction=Number(scenario.suctionCode||scenario.suctionMaxCode||0);
  const mode=String(scenario.suctionMode||"");
  const baseDirt=Number(baseScenario.dirtCode||baseScenario.dirtMaxCode||0);
  const baseObs=Number(baseScenario.obstacleLevelCode||0);

  if(choice==="dust"){
    if(dirtMax>=3 || dirt>=3 || String(scenario.dirtLevel||"").includes("높"))return 100;
    if(dirt>=2)return 45;
    return -20;
  }
  if(choice==="pet"){
    let score=0;
    if(dirtMax>=3 || dirt>=3 || String(scenario.dirtLevel||"").includes("높"))score+=60;
    if(suction>=3 || mode.includes("강") || mode.includes("터보"))score+=60;
    return score || -20;
  }
  if(choice==="obstacle"){
    if(obs>=3 || String(scenario.obstacleLevel||"").includes("높"))return 100;
    if(obs>=2 || String(scenario.obstacleLevel||"").includes("중"))return 45;
    return -20;
  }
  // 평소와 같음은 1회차 학습 프로필의 오염도/장애물 수준과 가까운 후보를 우선합니다.
  return 70 - Math.abs((dirt||baseDirt)-baseDirt)*12 - Math.abs((obs||baseObs)-baseObs)*12;
}

function scenarioPoolForChoices(choices){
  if(!predictionData.runs || !predictionData.runs.length)return [];
  if(choices.scope==="home"){
    return predictionData.runs.map(r=>r.home).filter(Boolean);
  }
  const zoneNo=Number(choices.zoneNumber||1);
  const pool=[];
  predictionData.runs.forEach(r=>{
    const z=(r.zones||[]).find(item=>Number(item.zone)===zoneNo);
    if(z)pool.push(z);
  });
  return pool;
}

function scoreScenarioForChoices(scenario,choices,baseScenario){
  let score=0;
  score+=cleanModeScore(scenario,choices.cleanMode);
  score+=suctionPreferenceScore(scenario,choices.intensity);
  score+=todayStateScore(scenario,choices.todayState,baseScenario);

  // 1회차 학습한 우리 집 정보과 최대한 가까운 후보를 우선합니다.
  if(Number(scenario.areaPyung)===Number(activeRun.home.areaPyung))score+=180;
  else score-=Math.abs(Number(scenario.areaPyung||0)-Number(activeRun.home.areaPyung||0))*4;

  if(choices.scope==="zone"){
    if(Number(scenario.zone)===Number(choices.zoneNumber))score+=80;
    if(baseScenario.floorType && scenario.floorType===baseScenario.floorType)score+=90;
    else if(baseScenario.floorType && scenario.floorType)score-=35;
    score-=Math.abs(Number(scenario.cleaningAreaM2||0)-Number(baseScenario.cleaningAreaM2||0))*1.5;
  }else{
    score-=Math.abs(Number(scenario.cleaningAreaM2||0)-Number(activeRun.home.cleaningAreaM2||0))*.25;
  }

  // 동점 방지를 위한 아주 작은 랜덤값. 조건 점수 자체에는 영향이 거의 없습니다.
  score+=Math.random()*0.01;
  return score;
}

function findMlScenarioFromChoices(choices){
  const baseScenario=getBaseScenarioFromChoices(choices);
  const pool=scenarioPoolForChoices(choices);
  if(!pool.length){
    const fallback=Object.assign({},baseScenario);
    fallback.matchNote="저장된 우리 집 기록으로 준비";
    return fallback;
  }

  let best=null;
  let bestScore=-Infinity;
  pool.forEach(candidate=>{
    const score=scoreScenarioForChoices(candidate,choices,baseScenario);
    if(score>bestScore){bestScore=score;best=candidate;}
  });

  const scenario=Object.assign({},best||baseScenario);
  scenario.scope=choices.scope;
  if(choices.scope==="zone"){
    scenario.zone=Number(choices.zoneNumber||scenario.zone||1);
    scenario.label=scenario.zone+"구역";
  }else{
    scenario.label="집 전체";
  }
  scenario.cleanModeChoice=choices.cleanMode;
  scenario.cleanModeLabel=cleanModeLabels[choices.cleanMode]||scenario.cleaningType;
  scenario.intensityChoice=choices.intensity;
  scenario.intensityLabel=intensityLabels[choices.intensity]||"표준";
  scenario.todayStateChoice=choices.todayState;
  scenario.todayStateLabel=todayStateLabels[choices.todayState]||"평소와 같음";
  scenario.targetSoc=targetFromRequired(scenario.requiredSoc);
  scenario.matchNote="오늘 상태에 맞춰 로보킹이 준비";
  scenario.matchBasis="청소 방식·오염도·장애물 상태 반영";

  if(choices.cleanMode==="both" && getCleanModeCandidateValue(scenario)!=="both"){
    scenario.matchNote="비슷한 청소 기록으로 준비";
  }
  return scenario;
}

function syncScenarioToState(scenario){
  state.selectedScenario=scenario;
  state.selectedScope=scenario.scope;
  state.selectedZone=scenario.scope==="zone"?scenario.zone:null;
  state.selectedLabel=scenario.label;
  state.requiredSoc=Number(scenario.requiredSoc);
  state.targetSoc=Number(scenario.targetSoc);
  state.modelName=scenario.modelName;
  state.globalRunId=scenario.globalRunId;
  state.areaPyung=scenario.areaPyung;
  state.cleaningAreaM2=scenario.cleaningAreaM2;
  state.cleaningType=scenario.cleaningType;
  state.mopEnabled=Boolean(scenario.mopEnabled);
  state.obstacleLevel=scenario.obstacleLevel;
  state.floorType=scenario.floorType;
  state.dirtLevel=scenario.dirtLevel;
  state.suctionMode=scenario.suctionMode;
  state.cleanModeChoice=scenario.cleanModeChoice||state.cleanModeChoice;
  state.cleanModeLabel=scenario.cleanModeLabel||scenario.cleaningType||state.cleaningType;
  state.intensityChoice=scenario.intensityChoice||state.intensityChoice;
  state.intensityLabel=scenario.intensityLabel||state.intensityLabel;
  state.todayStateChoice=scenario.todayStateChoice||state.todayStateChoice;
  state.todayStateLabel=scenario.todayStateLabel||state.todayStateLabel;
  state.matchNote=scenario.matchNote||"우리 집 기록으로 맞춤 준비";
  state.matchBasis=scenario.matchBasis||"오늘 상태 반영";
  state.cleaningRemainingSoc=Number(state.requiredSoc||0);
  state.cleaningSegmentIndex=0;
  state.splitCleaning=state.requiredSoc>MAX_SINGLE_PASS_USE;
  state.progress=0;
  state.missionDone=false;
  state.area=scenario.cleaningAreaM2||state.area;
}

function getManualSelectionKey(){
  const c=getPredictionChoices();
  return [c.scopeValue,c.cleanMode,c.intensity,c.todayState].join("|");
}

function predictSocFromConditions(autoExecuteAfter=false){
  if(state.cleaning||state.charging||state.mapping){showToast("학습/청소/충전이 끝난 뒤 다시 준비할 수 있어요.");return}

  if(!state.profileReady){
    setGuide("아직 로보킹이 우리 집을 잘 몰라요. 먼저 1회차 학습 청소를 시작해 주세요.","warning");
    showToast("먼저 로보킹에게 우리 집을 알려주세요.");
    $("speech").innerHTML="<strong style='color:#ef8c32'>아직 학습 전이에요!</strong><br>먼저 우리 집을 알려주세요.";
    setModeChipText("🏠 1회차 학습 필요");
    switchPage("homePage");
    return;
  }
  const choices=getPredictionChoices();
  const currentManualKey=getManualSelectionKey();
  const matchedScenario=findMlScenarioFromChoices(choices);
  const loading=$('predictLoading');
  state.predicting=true;
  if(loading){loading.textContent="로보킹이 오늘 청소를 준비하고 있어요...";loading.classList.add('active');}
  $("speech").innerHTML="<strong style='color:#2f8b3a'>잠깐만요!</strong><br>오늘 청소 준비를 하고 있어요.";
  setModeChipText("🤖 우리 집 기록으로 준비 중");
  setGuide("오늘 상태를 보고 로보킹이 청소 준비를 하고 있어요. 잠시만 기다려 주세요.","charging");
  showToast("청소 준비 중: 오늘 상태에 맞춰 준비하고 있어요.");

  setTimeout(()=>{
    syncScenarioToState(matchedScenario);
    state.smartCleanMode="manual";
    state.mapMode="view";
    state.selectedDirtyZones=[];
    state.cleaningZones=getCleaningZonesForCurrentPlan();
    state.completedZones=[];
    state.currentCleaningZone=null;
    state.manualReady=true;
    state.manualKey=currentManualKey;
    state.predicted=true;
    state.predicting=false;
    state.chargeComplete=false;
    if(loading){
      const status=state.soc>=state.targetSoc?"바로 청소 가능":"충전 필요";
      loading.textContent="준비 완료 · "+status+" · 로보킹이 필요한 만큼 준비했어요.";
      loading.classList.remove('active');
    }
    render();
    const statusText=state.soc>=state.targetSoc?"바로 청소할 수 있어요":"잠깐 충전하면 청소할 수 있어요";
    $("speech").innerHTML="<strong style='color:#2f8b3a'>준비 완료!</strong><br>"+statusText;
    setModeChipText("✅ 청소 준비 완료 · "+state.selectedLabel);
    addEvent("청소 준비 완료",state.selectedLabel+" 청소를 위해 필요한 배터리 "+state.targetSoc+"%만 준비했어요.","배터리 절약");
    setGuide(statusText.includes("바로")?"준비 완료! 바로 출동할게요.":"준비 완료! 필요한 만큼만 채우고 바로 출발할게요.", state.soc>=state.targetSoc?"done":"warning");
    showToast("청소 준비 완료! 로보킹이 오늘 청소 준비를 마쳤어요.");
    if(autoExecuteAfter){
      setTimeout(()=>executeTopClean(),260);
    }
  },900);
}

function floorSummary(){
  if(!activeRun || !activeRun.zones)return "정보 없음";
  const counts={};
  activeRun.zones.forEach(z=>{const k=z.floorType||"정보 없음";counts[k]=(counts[k]||0)+1;});
  return Object.keys(counts).map(k=>k+" "+counts[k]+"구역").join(", ");
}
function floorKindCount(){
  if(!activeRun || !activeRun.zones)return 0;
  const kinds={};
  activeRun.zones.forEach(z=>{kinds[z.floorType||"정보 없음"]=true;});
  return Object.keys(kinds).length;
}
function dirtSummaryShort(){
  if(!activeRun || !activeRun.zones)return "정보 없음";
  const high=activeRun.zones.filter(z=>String(z.dirtLevel||"").includes("높")).length;
  const mid=activeRun.zones.filter(z=>String(z.dirtLevel||"").includes("중") || String(z.dirtLevel||"").includes("보통")).length;
  if(high>0)return "높음 "+high+"구역"+(mid>0?" · 보통 "+mid+"구역":"");
  if(mid>0)return "보통 "+mid+"구역";
  return dirtSummary();
}
function dirtSummary(){
  if(!activeRun || !activeRun.zones)return "정보 없음";
  const counts={};
  activeRun.zones.forEach(z=>{const k=z.dirtLevel||"정보 없음";counts[k]=(counts[k]||0)+1;});
  return Object.keys(counts).map(k=>k+" "+counts[k]+"구역").join(", ");
}
function obstacleSummary(){return activeRun && activeRun.home ? (activeRun.home.obstacleLevel||"중간") : "중간";}
function profileResultBody(){
  const startSoc=Number(state.firstRunStartSoc||0);
  const endSoc=Number(state.firstRunEndSoc||state.soc||0);
  const recorded=Number(state.firstRunSocUsed||state.firstRunRequiredSoc||0);
  const used=Number(state.firstRunSocUsed||0);
  const socLine=state.firstRunSocEnough
    ? "배터리 변화: <b>"+Math.round(startSoc)+"% → "+Math.round(endSoc)+"%</b>"
    : "배터리 변화: <b>"+Math.round(startSoc)+"% → "+Math.round(endSoc)+"%</b> <small>(배터리 부족)</small>";
  return "<b>우리 집 저장 완료</b><br><br>"
    +"집 크기: <b>"+activeRun.areaPyung+"평 · "+activeRun.home.cleaningAreaM2+"㎡</b><br>"
    +"구역: <b>"+getDisplayZoneCount()+"개</b><br>"
    +"바닥: <b>"+floorKindCount()+"종 혼합</b><br>"
    +"오염도: <b>"+dirtSummaryShort()+"</b><br>"
    +"장애물: <b>"+obstacleSummary()+"</b><br>"
    +"학습 중 사용한 배터리: <b>"+fmtSoc(recorded)+"%</b><br>"
    +socLine+"<br>"
    +"현재 배터리: <b>"+Math.round(state.soc)+"%</b><br><br>"
    +"다음 단계: <b>오늘 청소 준비하기</b>";
}
function startFirstMapping(){
  state.chargePurpose='current';state.nextHomeReady=false;
  if(state.cleaning||state.charging||state.mapping){showToast("진행 중인 작업이 끝난 뒤 다시 시도해 주세요.");return}

  const startSoc=clamp(Math.round(Number(state.soc||0)),0,100);
  if(startSoc < MIN_SOC_AFTER_LEARNING + MIN_LEARNING_SOC_USE){
    openModal("학습 전에 잠깐 충전할게요","처음 우리 집을 배우려면<br>로보킹에게 힘이 조금 더 필요해요.<br><br>잠깐 충전한 뒤 시작하면<br>집 구조를 더 안정적으로 배울 수 있어요.");
    return;
  }

  // 현재 배터리 기준으로 학습 주행 후 여유 배터리를 남길 수 있는 기록 시나리오를 우선 선택합니다.
  const safeRuns=predictionData.runs.filter(r=>getLearningSocUse(r,startSoc)>0);
  activeRun=pickRandomRun(safeRuns.length?safeRuns:predictionData.runs) || predictionData.runs[0];
  syncScenarioToState(activeRun.home);
  refreshScopeSelect();

  const fullRequiredSoc=Number(activeRun.home.requiredSoc||0);
  const learningUse=getLearningSocUse(activeRun,startSoc);
  const expectedEndSoc=Math.max(MIN_SOC_AFTER_LEARNING,startSoc-learningUse);

  state.profileReady=false;
  state.predicted=false;
  state.mapping=true;
  state.mappingProgress=0;
  state.mappingStepIndex=0;
  state.firstRunFullRequiredSoc=fullRequiredSoc;
  state.firstRunRequiredSoc=learningUse;
  state.firstRunSocUsed=learningUse;
  state.firstRunStartSoc=startSoc;
  state.firstRunEndSoc=expectedEndSoc;
  state.firstRunSocEnough=true;
  state.soc=startSoc;
  state.chargeComplete=false;
  state.celebrating=false;
  switchPage("homePage");
  setGuide("학습 청소를 시작했어요. 로보킹이 우리 집 구조와 바닥 상태를 차근차근 기억하고 있어요.","charging");
  showToast("학습 시작: 로보킹이 우리 집을 배우고 있어요.");
  render();

  let tick=0;
  const total=mappingSteps.length*4;
  const timer=setInterval(()=>{
    tick+=1;
    const ratio=Math.min(1,tick/total);
    state.mappingProgress=Math.min(100,Math.round(ratio*100));
    state.mappingStepIndex=Math.min(mappingSteps.length-1,Math.floor((tick-1)/4));
    state.progress=state.mappingProgress;
    state.soc=Math.max(MIN_SOC_AFTER_LEARNING,Math.round((startSoc-learningUse*ratio)*10)/10);
    state.firstRunEndSoc=state.soc;
    state.temperature=Math.min(33,state.temperature+0.08);
    render();
    if(tick>=total){
      clearInterval(timer);
      state.mapping=false;
      state.profileReady=true;
      state.predicted=false;
      state.progress=100;
      state.soc=Math.max(MIN_SOC_AFTER_LEARNING,Math.round(expectedEndSoc));
      state.firstRunEndSoc=state.soc;
      const scopeSelect=$("scopeSelect"); if(scopeSelect)scopeSelect.value='home';
      const cleanModeSelect=$("cleanModeSelect"); if(cleanModeSelect)cleanModeSelect.value=activeRun.home.mopEnabled?'mop':'dry';
      const intensitySelect=$("intensitySelect"); if(intensitySelect)intensitySelect.value='standard';
      const todayStateSelect=$("todayStateSelect"); if(todayStateSelect)todayStateSelect.value='normal';
      state.temperature=29;
      state.learnCount=(state.learnCount||0)+1;
      const eventMsg="집 구조와 바닥 상태를 기억했어요. 학습에 배터리 "+fmtSoc(learningUse)+"%만 사용하고 15% 이상 남겼어요.";
      addEvent("1회차 학습 청소 완료",eventMsg,"집 정보 저장");
      spawnEffect("🏠",8);spawnEffect("✨",9);
      render();
      setGuide("우리 집 저장 완료! 이제 오늘 청소 조건을 고르면 로보킹이 알아서 준비해요.","done");
      setTimeout(()=>{render();checkMissionUnlock();},350);
    }
  },260);
}
function selectScenario(scope,zoneNumber=null){
  if(state.cleaning||state.charging||state.mapping){showToast("학습/청소/충전이 끝난 뒤 변경할 수 있어요.");return}
  if(!state.profileReady){
    showToast("구역 선택은 1회차 학습 후 가능해요.");
    $("speech").innerHTML="<strong style='color:#ef8c32'>우리 집 학습이 먼저예요</strong><br>1회차 학습 청소를 시작해 주세요.";
    return;
  }
  if(!state.predicted){setGuide("구역을 바꾸기 전 오늘 청소 준비를 먼저 완료해 주세요.","warning");showToast("먼저 오늘 청소 준비하기를 눌러주세요.");return}
  const scopeSelect=$('scopeSelect');
  if(scopeSelect)scopeSelect.value=scope==="home"?"home":String(zoneNumber||1);
  const choices=getPredictionChoices(scope,zoneNumber);
  const matchedScenario=findMlScenarioFromChoices(choices);
  syncScenarioToState(matchedScenario);
  render();
  const status=state.soc>=state.targetSoc?"청소 가능":"충전 필요";
  const loading=$('predictLoading');
  if(loading)loading.textContent=state.selectedLabel+" 선택 · "+status+" · 로보킹이 다시 준비했어요.";
  $("speech").innerHTML="<strong>"+state.selectedLabel+" 선택!</strong><br>이 구역에 맞춰 다시 준비했어요.";
  setModeChipText("✨ "+state.selectedLabel+" 청소 준비 완료");
  setGuide((state.soc>=state.targetSoc)?state.selectedLabel+" 청소 준비가 끝났어요. 지금 바로 출동할 수 있어요.":state.selectedLabel+" 청소 준비가 끝났어요. 잠깐 충전하고 출발하면 좋아요.", state.soc>=state.targetSoc?"done":"warning");
  showToast(state.selectedLabel+" 청소 준비를 다시 맞췄어요.");
}

function openScenarioModal(){
  const enough=state.soc>=state.targetSoc;
  const title=enough?"바로 출동할 수 있어요!":"조금만 더 힘을 채울게요!";
  const scopeText=state.selectedScope==="home"?"집 전체 청소":state.selectedLabel+" 청소";
  const conditionLine="오늘 조건: <b>"+state.cleanModeLabel+" · "+state.intensityLabel+" · "+state.todayStateLabel+"</b><br><br>";
  const zoneLine=state.selectedScope==="zone"
    ? state.selectedLabel+"은 <b>"+(state.floorType||"바닥 정보")+"</b> 바닥이고, 오늘은 <b>"+(state.dirtLevel||"평소")+"</b> 상태로 준비했어요.<br><br>"
    : "저장해둔 우리 집 정보를 바탕으로 집 전체 청소를 준비했어요.<br><br>";
  const body=enough
    ? scopeText+" 준비가 끝났어요.<br><br>"+conditionLine+zoneLine+"지금 바로 시작해도 충분해요.<br>제가 배터리를 아끼면서 청소할게요!"
    : scopeText+" 준비가 끝났어요.<br><br>"+conditionLine+zoneLine+"지금 바로 출발하기엔 힘이 조금 부족해요.<br><br><b>잠깐 충전하고 나면</b><br>더 편하게 청소를 마칠 수 있어요.";
  if(enough){
    openModal(title,body);
  }else{
    openModal(title,body,{showCancel:true,cancelText:"취소",confirmText:"충전하고 시작",onConfirm:()=>chargeRobot(false)});
  }
}


function getHomeSizeType(areaPyung){
  const zoneCount=(activeRun && Number(activeRun.zoneCount)) || (activeRun && activeRun.home && Number(activeRun.home.zoneCount)) || 0;
  if(zoneCount===4)return "small";
  if(zoneCount===6)return "medium";
  if(zoneCount===8)return "large";
  const area=Number(areaPyung||0);
  if(area<=24)return "small";
  if(area<=49)return "medium";
  return "large";
}
function getHomeSizeLabel(areaPyung){
  const type=getHomeSizeType(areaPyung);
  if(type==="small")return "소형";
  if(type==="medium")return "중형";
  return "대형";
}
function getExpectedZoneCount(areaPyung){
  if(activeRun && Number(activeRun.zoneCount))return Number(activeRun.zoneCount);
  if(activeRun && activeRun.home && Number(activeRun.home.zoneCount))return Number(activeRun.home.zoneCount);
  const type=getHomeSizeType(areaPyung);
  if(type==="small")return 4;
  if(type==="medium")return 6;
  return 8;
}
function getActualZoneCount(){
  return (activeRun && activeRun.zones && activeRun.zones.length) ? activeRun.zones.length : 0;
}
function getDisplayZoneCount(){
  const actual=getActualZoneCount();
  if(actual)return actual;
  const area=(activeRun && activeRun.areaPyung) ? activeRun.areaPyung : state.areaPyung;
  return getExpectedZoneCount(area);
}
function getZoneByNumber(zoneNo){
  const zones=(activeRun && activeRun.zones) ? activeRun.zones : [];
  return zones.find(z=>Number(z.zone)===Number(zoneNo)) || zones[zoneNo-1] || null;
}
function normalizeDirtCode(zone){
  if(!zone)return 2;
  const code=Number(zone.dirtCode || zone.dirtLevelCode || 0);
  if(Number.isFinite(code) && code>0)return code;

  const text=String(zone.dirtLevel || zone.dirt || "").toLowerCase();
  if(text.includes("매우") || text.includes("심함") || text.includes("높") || text.includes("heavy") || text.includes("high"))return 4;
  if(text.includes("중") || text.includes("보통") || text.includes("normal") || text.includes("medium"))return 2;
  if(text.includes("낮") || text.includes("깨끗") || text.includes("low") || text.includes("clean"))return 1;
  return 2;
}
function getCleanableZones(){
  const blocked=state.noGoZones || [];
  const zones=(activeRun && activeRun.zones) ? activeRun.zones : [];
  return zones.filter(z=>!blocked.includes(Number(z.zone)));
}
function getAllCleanableZoneNumbers(){
  return getCleanableZones().map(z=>Number(z.zone)).filter(n=>Number.isFinite(n));
}
function getWholeHomeZones(){
  if(activeRun && activeRun.zones && activeRun.zones.length){
    return activeRun.zones.slice().sort((a,b)=>Number(a.zone)-Number(b.zone));
  }
  return [];
}
function getWholeHomeRequiredSoc(){
  // 최종 ML 구조 기준:
  // 집 전체 필요 SOC = 소형 4개 / 중형 6개 / 대형 8개 zone의 requiredSoc 합산
  const zones=getWholeHomeZones();
  if(zones.length){
    return Math.round(zones.reduce((sum,z)=>sum+Number(z.requiredSoc||0),0)*10)/10;
  }
  if(activeRun && activeRun.home){
    return Math.round(Number(activeRun.home.requiredSoc||0)*10)/10;
  }
  return 0;
}
function getWholeHomeTargetSoc(){
  // 집 전체 청소 가능하도록 필요한 SOC + 최소 잔량 15%, 최대 90% 상한 적용
  return targetFromRequired(getWholeHomeRequiredSoc());
}
function getPlannedZoneNumbers(){
  const blocked=state.noGoZones || [];

  // AI 자동청소는 반드시 '금지구역을 제외한 전체 구역'입니다.
  // 이전에 '더러운 곳만'이나 특정 구역을 눌렀던 흔적이 cleaningZones에 남아도
  // AI 자동청소에서는 그 값을 무시합니다.
  if(state.smartCleanMode==="auto"){
    return getAllCleanableZoneNumbers();
  }

  // 더러운 곳만은 오염도가 높은 일부 구역만 선택합니다.
  if(state.smartCleanMode==="dirty" && state.selectedDirtyZones && state.selectedDirtyZones.length){
    return state.selectedDirtyZones.filter(n=>!blocked.includes(Number(n))).map(Number);
  }

  // 사용자가 특정 구역을 직접 선택한 경우
  if(state.smartCleanMode==="zone" && state.selectedScope==="zone" && state.selectedZone){
    return blocked.includes(Number(state.selectedZone)) ? [] : [Number(state.selectedZone)];
  }

  // 실제 청소 진행 중에는 시작 시 확정한 청소 구역을 사용합니다.
  if(state.cleaningZones && state.cleaningZones.length){
    return state.cleaningZones.filter(n=>!blocked.includes(Number(n))).map(Number);
  }

  return getAllCleanableZoneNumbers();
}
function getDirtyRecommendationZones(){
  const cleanable=getCleanableZones();
  if(!cleanable.length)return [];
  const sorted=cleanable.slice().sort((a,b)=>getZoneConditionScore(b.zone)-getZoneConditionScore(a.zone));
  const count=Math.min(Math.max(1,Math.ceil(sorted.length*0.35)),3);
  return sorted.slice(0,count).sort((a,b)=>Number(a.zone)-Number(b.zone)).map(z=>Number(z.zone));
}
function getCurrentRecommendation(){
  if(!state.profileReady)return {icon:"🏠",title:"먼저 우리 집을 알려주세요",sub:"1회차 학습 후 맞춤 청소를 추천해요."};
  const dirty=getDirtyRecommendationZones();
  const noGo=(state.noGoZones||[]).length;
  if(dirty.length>=2)return {icon:"🔥",title:"더러운 곳만 먼저 해볼까요?",sub:dirty.join(", ")+"번 영역을 빠르게 청소할 수 있어요."};
  if(noGo>0)return {icon:"🚫",title:"금지구역은 조용히 지나갈게요",sub:"설정한 "+noGo+"곳은 빼고 청소해요."};
  return {icon:"✨",title:"AI 자동청소가 좋아요",sub:"로보킹이 오늘 상태에 맞춰 알아서 준비해요."};
}

function getZoneConditionScore(zoneNo){
  const zone=getZoneByNumber(zoneNo);
  if(!zone){
    // CSV에 아직 해당 영역 데이터가 없을 때도 화면이 비어 보이지 않도록 부드럽게 분산
    return Number(zoneNo||1) * 0.35;
  }

  const dirt=normalizeDirtCode(zone);
  const suction=Number(zone.suctionCode || zone.suctionModeCode || 0);
  const required=Number(zone.requiredSoc || 0);
  const obstacle=Number(zone.obstacleLevelCode || 0);

  // 바닥 상태 색상은 오염도 중심으로 보되,
  // 흡입 강도/장애물/필요 배터리를 조금 섞어서 구역별 차이가 잘 보이게 합니다.
  return dirt*10 + suction*2 + obstacle*1.2 + required*0.18 + Number(zoneNo||1)*0.03;
}
function getDirtVisual(zoneNo){
  const count=getDisplayZoneCount();
  const scores=[];
  for(let i=1;i<=count;i++){
    scores.push({zone:i,score:getZoneConditionScore(i)});
  }
  scores.sort((a,b)=>a.score-b.score || a.zone-b.zone);
  const rank=Math.max(0,scores.findIndex(s=>Number(s.zone)===Number(zoneNo)));
  const ratio=(rank+1)/Math.max(scores.length,1);

  // 사용자에게는 '오염도 수치'가 아니라 같은 집 안에서 상대적으로 더 신경쓸 구역을 색으로 보여줍니다.
  // 그래서 데이터가 전부 낮음/보통에 몰려도 맵에서는 구역 차이가 한눈에 보이도록 색을 넓게 분산합니다.
  if(ratio<=0.25)return {fill:"#cfeec0", label:"깨끗"};
  if(ratio<=0.55)return {fill:"#ffe08a", label:"보통"};
  if(ratio<=0.82)return {fill:"#ffb169", label:"먼지"};
  return {fill:"#ff7d68", label:"집중"};
}
function getMapZoneClass(zoneNo){
  const n=Number(zoneNo);
  const classes=[];
  const noGo=state.noGoZones || [];
  const planned=getPlannedZoneNumbers();
  const selected=state.selectedDirtyZones || [];
  const completed=state.completedZones || [];
  const dirtyHighlightOn = state.smartCleanMode==="dirty" && state.mapMode!=="noGo";

  if(noGo.includes(n))classes.push("no-go");
  else{
    if(planned.includes(n))classes.push("planned");
    if(dirtyHighlightOn && selected.includes(n))classes.push("dirty-selected");
    if(dirtyHighlightOn && selected.length && !selected.includes(n))classes.push("dimmed");
    if(state.cleaning && Number(state.currentCleaningZone)===n)classes.push("cleaning-zone");
    if(completed.includes(n))classes.push("completed");
  }
  return classes.join(" ");
}
function routeClass(){
  // 선택 직후 지도 위에 경로선이 지나가면 오류처럼 보여서,
  // 경로선은 실제 청소가 진행될 때만 표시합니다.
  const active=Boolean(state.cleaning);
  return "map-route"+(active?" active-route":"");
}
function mapRoom(x,y,w,h,rx,zoneNo,label,dashed=false){
  const visual=getDirtVisual(zoneNo);
  const isNoGo=state.noGoZones && state.noGoZones.includes(Number(zoneNo));
  const zoneClass=getMapZoneClass(zoneNo);

  const compact = h < 42 || w < 66;
  const labelSize = compact ? 11.6 : 13.0;
  const labelY = compact ? y + h * 0.57 : y + h * 0.54;
  const badgeR = compact ? 8 : 9;
  const badgeX = x + w - badgeR - 6;
  const badgeY = y + badgeR + 6;
  const badgeFont = compact ? 7.8 : 8.8;
  const centerX=x+w/2;

  let html = "<g class='map-room-group "+zoneClass+"' data-action='mapZone' data-zone='"+zoneNo+"'>"
    +"<rect class='map-room"+(dashed?" dashed":"")+"' x='"+x+"' y='"+y+"' width='"+w+"' height='"+h+"' rx='"+rx+"' fill='"+visual.fill+"'></rect>";

  if(isNoGo){
    html += "<rect class='map-no-go-shade' x='"+x+"' y='"+y+"' width='"+w+"' height='"+h+"' rx='"+rx+"'></rect>"
      +"<line class='map-no-go-line' x1='"+(x+10)+"' y1='"+(y+10)+"' x2='"+(x+w-10)+"' y2='"+(y+h-10)+"'></line>"
      +"<line class='map-no-go-line' x1='"+(x+w-10)+"' y1='"+(y+10)+"' x2='"+(x+10)+"' y2='"+(y+h-10)+"'></line>";
  }else if(state.smartCleanMode==="dirty" && state.mapMode!=="noGo" && (state.selectedDirtyZones||[]).includes(Number(zoneNo))){
    // 더러운 곳만 모드에서는 별도 아이콘 없이 초록색 테두리만 깜빡이게 표시합니다.
  }

  html += "<circle cx='"+badgeX+"' cy='"+badgeY+"' r='"+badgeR+"' fill='rgba(255,255,255,.82)'></circle>"
    +"<text class='map-room-sub' style='font-size:"+badgeFont+"px' x='"+badgeX+"' y='"+(badgeY+0.5)+"'>"+zoneNo+"</text>"
    +"<text class='map-room-label' style='font-size:"+labelSize+"px' x='"+centerX+"' y='"+labelY+"'>"+(isNoGo?"금지":label)+"</text>";

  if(!isNoGo && (state.completedZones||[]).includes(Number(zoneNo))){
    html += "<text class='map-check' x='"+(x+12)+"' y='"+(y+13)+"'>✓</text>";
  }
  html += "</g>";

  return html;
}
// 지도 레이아웃(좌표)을 한곳에 모아, 방 그리기와 로봇 위치 계산이 같은 좌표를 쓰도록 합니다.
const MAP_LAYOUTS={
  small:{viewBox:"0 0 244 162",label:"소형 집 구조 맵",route:"M42 48 C94 48, 112 96, 176 108",
    rooms:[[18,18,112,54,12,1,"거실"],[134,18,92,54,12,2,"주방"],[18,76,130,68,12,3,"침실"],[152,76,74,68,12,4,"현관",true]]},
  medium:{viewBox:"0 0 236 174",label:"중형 집 구조 맵",route:"M48 42 C96 52, 114 94, 182 98 C170 128, 118 138, 64 146",
    rooms:[[14,14,92,50,12,1,"침실"],[110,14,112,50,12,2,"주방"],[14,68,124,58,12,3,"거실"],[142,68,80,58,12,4,"카펫",true],[14,130,92,32,10,5,"현관"],[110,130,112,32,10,6,"다용도"]]},
  large:{viewBox:"0 0 246 172",label:"대형 집 구조 맵",route:"M46 34 C96 44, 146 36, 202 36 C174 76, 162 98, 210 92 C166 126, 102 142, 54 140",
    rooms:[[12,12,72,44,11,1,"침실1"],[88,12,72,44,11,2,"침실2"],[164,12,70,44,11,3,"주방"],[12,60,106,58,12,4,"거실"],[122,60,58,58,12,5,"현관"],[184,60,50,58,12,6,"카펫",true],[12,122,106,38,10,7,"서재"],[122,122,112,38,10,8,"다용도"]]}
};
function currentMapLayout(){
  const area=(activeRun && activeRun.areaPyung) ? activeRun.areaPyung : state.areaPyung;
  return MAP_LAYOUTS[getHomeSizeType(area)]||MAP_LAYOUTS.large;
}
function getZoneRect(zoneNo){
  const r=currentMapLayout().rooms.find(x=>Number(x[5])===Number(zoneNo));
  return r?{x:r[0],y:r[1],w:r[2],h:r[3]}:null;
}
// 방 안을 지그재그(잔디깎기)로 훑는 경로 꼭짓점들
function zonePathPoints(rect){
  const m=11;
  const x0=rect.x+m,x1=rect.x+rect.w-m,y0=rect.y+m,y1=rect.y+rect.h-m;
  const rows=Math.max(2,Math.round((y1-y0)/11)+1);
  const pts=[];
  for(let r=0;r<rows;r++){
    const y=y0+(y1-y0)*r/(rows-1);
    const ltr=r%2===0;
    pts.push([ltr?x0:x1,y]);pts.push([ltr?x1:x0,y]);
  }
  return pts;
}
function pointAlong(pts,t){
  const segs=[];let total=0;
  for(let i=1;i<pts.length;i++){const d=Math.hypot(pts[i][0]-pts[i-1][0],pts[i][1]-pts[i-1][1]);segs.push(d);total+=d;}
  let target=clamp(t,0,1)*total;
  for(let i=0;i<segs.length;i++){
    if(target<=segs[i]||i===segs.length-1){
      const k=segs[i]?Math.min(1,target/segs[i]):1;
      return {x:pts[i][0]+(pts[i+1][0]-pts[i][0])*k,y:pts[i][1]+(pts[i+1][1]-pts[i][1])*k,dir:Math.sign(pts[i+1][0]-pts[i][0])||1,idx:i};
    }
    target-=segs[i];
  }
  return {x:pts[0][0],y:pts[0][1],dir:1,idx:0};
}
// 화면 진행률(state.progress)은 320ms 단위로 끊겨서, 로봇 아이콘은 시간 기반의 연속 진행률로 움직입니다.
function getContinuousCleanProgress(){
  const a=state.cleanAnim;
  if(!a)return Number(state.progress||0);
  const e=clamp((Date.now()-a.startedAt)/a.duration,0,1);
  return Math.min(a.fromProgress+(a.toProgress-a.fromProgress)*e,99.9);
}
function getMapRobotPose(){
  if(!state.cleaning)return null;
  const zones=(state.cleaningZones&&state.cleaningZones.length)?state.cleaningZones:getCleaningZonesForCurrentPlan();
  if(!zones.length)return null;
  const p=getContinuousCleanProgress()/100*zones.length;
  const idx=Math.min(zones.length-1,Math.floor(p));
  const sub=p-idx;
  const rect=getZoneRect(zones[idx]);
  if(!rect)return null;
  const pts=zonePathPoints(rect);
  const TRAVEL=0.16;
  if(idx>0 && sub<TRAVEL){
    const prevRect=getZoneRect(zones[idx-1]);
    const prevPts=prevRect?zonePathPoints(prevRect):pts;
    const from=prevPts[prevPts.length-1],to=pts[0];
    const k=sub/TRAVEL;
    return {x:from[0]+(to[0]-from[0])*k,y:from[1]+(to[1]-from[1])*k,dir:Math.sign(to[0]-from[0])||1,sweep:"",zone:zones[idx],idx:idx};
  }
  const t=idx>0?(sub-TRAVEL)/(1-TRAVEL):sub;
  const pos=pointAlong(pts,t);
  const swept=pts.slice(0,pos.idx+1).concat([[pos.x,pos.y]]);
  const d="M"+swept.map(q=>q[0].toFixed(1)+" "+q[1].toFixed(1)).join(" L");
  return {x:pos.x,y:pos.y,dir:pos.dir,sweep:d,zone:zones[idx],idx:idx};
}
function robotTransform(pose){
  return "translate("+pose.x.toFixed(1)+" "+pose.y.toFixed(1)+") scale("+(pose.dir<0?-1:1)+" 1)";
}
function getMapRobotMarkup(){
  const pose=getMapRobotPose();
  if(!pose)return "";
  return "<path id='mapSweep' class='map-sweep' d='"+pose.sweep+"'></path>"
    +"<g id='mapRobot' class='map-robot' transform='"+robotTransform(pose)+"'>"
    +"<circle class='map-robot-puff' cx='-9' cy='3' r='2'></circle><circle class='map-robot-puff p2' cx='-10' cy='0' r='1.6'></circle><circle class='map-robot-puff p3' cx='-8' cy='-3' r='1.3'></circle>"
    +"<ellipse class='map-robot-shadow' cx='0' cy='7.5' rx='8' ry='2.4'></ellipse>"
    +"<g class='map-robot-body'>"
    +"<circle r='8.5' class='map-robot-shell'></circle>"
    +"<rect x='-5.5' y='-0.5' width='11' height='5.5' rx='2.6' class='map-robot-face'></rect>"
    +"<circle cx='-2.6' cy='2.2' r='1' class='map-robot-eye'></circle><circle cx='2.6' cy='2.2' r='1' class='map-robot-eye'></circle>"
    +"<circle cx='0' cy='-4.6' r='1.5' class='map-robot-light'></circle>"
    +"<text class='map-robot-crown' x='0' y='-9'>👑</text>"
    +"</g></g>";
}
let mapRobotRaf=null;
function tickMapRobot(){
  mapRobotRaf=null;
  if(!state.cleaning)return;
  const pose=getMapRobotPose();
  if(pose){
    const g=$("mapRobot"),sw=$("mapSweep");
    if(g)g.setAttribute("transform",robotTransform(pose));
    if(sw)sw.setAttribute("d",pose.sweep);
    // 로봇이 다음 방으로 넘어간 순간 방 강조(초록 깜빡임/✓)도 같이 갱신
    if(Number(state.currentCleaningZone)!==Number(pose.zone)){
      const zones=(state.cleaningZones&&state.cleaningZones.length)?state.cleaningZones:getCleaningZonesForCurrentPlan();
      state.currentCleaningZone=pose.zone;
      state.completedZones=zones.slice(0,pose.idx);
      render();
    }
  }
  mapRobotRaf=requestAnimationFrame(tickMapRobot);
}
function startMapRobotAnim(){
  if(mapRobotRaf)cancelAnimationFrame(mapRobotRaf);
  mapRobotRaf=requestAnimationFrame(tickMapRobot);
}
function getMapSvg(type){
  const lay=MAP_LAYOUTS[type]||MAP_LAYOUTS.large;
  let rooms="";
  lay.rooms.forEach(r=>{rooms+=mapRoom(r[0],r[1],r[2],r[3],r[4],r[5],r[6],Boolean(r[7]));});
  return "<svg class='home-map-svg' viewBox='"+lay.viewBox+"' role='img' aria-label='"+lay.label+"'>"
    +"<path class='"+routeClass()+"' d='"+lay.route+"'></path>"
    +rooms
    +getMapRobotMarkup()
    +"</svg>";
}
function getDirtLegendHtml(){
  return "<div class='map-legend' aria-label='바닥 상태 색상 안내'>"
    +"<span class='map-legend-title'>바닥 상태</span>"
    +"<span class='map-legend-item'><i class='map-dot dot-clean'></i>깨끗</span>"
    +"<span class='map-legend-item'><i class='map-dot dot-normal'></i>보통</span>"
    +"<span class='map-legend-item'><i class='map-dot dot-dusty'></i>먼지</span>"
    +"<span class='map-legend-item'><i class='map-dot dot-focus'></i>집중</span>"
    +"</div>";
}
function getMapActionHtml(){
  const noGoCount=(state.noGoZones||[]).length;
  const noGoText=noGoCount>0 ? "금지 "+noGoCount : "금지구역";
  let hint="원하는 방식을 고른 뒤 청소하기를 누르면 돼요.";
  let readyClass=state.predicted?" status-ready":"";
  if(state.mapMode==="noGo")hint="지도에서 <b>청소하지 않을 영역</b>을 눌러주세요.";
  else if(noGoCount>0)hint="금지구역 "+noGoCount+"곳은 빼고 준비해요.";
  else if(state.smartCleanMode==="dirty")hint="초록 테두리 영역만 골라뒀어요. 청소하기를 누르면 그곳만 청소해요.";
  else if(state.smartCleanMode==="auto")hint="전체 영역을 모두 준비했어요. 금지구역만 빼고 청소해요.";

  return "<div class='map-action-row'>"
    +"<button class='map-action-btn"+(state.smartCleanMode==="auto" && state.mapMode!=="noGo"?" active":"")+"' data-action='aiAutoClean'>✨ AI 자동청소</button>"
    +"<button class='map-action-btn"+(state.smartCleanMode==="dirty" && state.mapMode!=="noGo"?" active":"")+"' data-action='dirtyOnlyClean'>🔥 더러운 곳만</button>"
    +"<button class='map-action-btn danger"+(state.mapMode==="noGo"?" active":"")+"' data-action='toggleNoGoMode'>🚫 "+noGoText+"</button>"
    +"</div>"
    +"<div class='map-action-hint"+readyClass+"'>"+hint+"</div>";
}
function getMapRecommendationHtml(){
  if(state.predicted || state.cleaning || state.charging)return "";
  const rec=getCurrentRecommendation();
  return "<div class='map-recommend-card'><span class='rec-icon'>"+rec.icon+"</span><div><b>"+rec.title+"</b><br>"+rec.sub+"</div></div>";
}
function getMapPrepCardHtml(){
  if(!state.predicted)return "";
  const planned=getPlannedZoneNumbers();
  const noGo=(state.noGoZones||[]).length;
  let title=state.selectedLabel+" 준비 완료";
  if(state.smartCleanMode==="dirty")title="더러운 곳만 준비 완료";
  if(state.smartCleanMode==="auto")title="AI 자동청소 준비 완료";
  const sub=(noGo>0?"금지구역 "+noGo+"곳 제외 · ":"")+(planned.length?planned.length+"개 영역 청소":"청소 영역 준비")+" · "+(state.soc<state.targetSoc?"충전 후 출발":"바로 출발 가능");
  const badge=state.soc<state.targetSoc?"충전 필요":"바로 가능";
  return "<div class='map-prep-card'><div><div class='map-prep-title'>"+title+"</div><div class='map-prep-sub'>"+sub+"</div></div><div class='map-prep-badge'>"+badge+"</div></div>";
}
function getLearnedMapHtml(){
  const area=(activeRun && activeRun.areaPyung) ? activeRun.areaPyung : state.areaPyung;
  const type=getHomeSizeType(area);
  return "<div class='home-map-card'>"
    +"<div class='home-map-img-wrap'>"+getMapSvg(type)+"</div>"
    +getDirtLegendHtml()
    +getMapRecommendationHtml()
    +getMapActionHtml()
    +getMapPrepCardHtml()
    +"</div>";
}

function refreshScopeSelect(){
  const scopeSelect=$('scopeSelect');
  if(!scopeSelect)return;

  const count=getDisplayZoneCount();
  const before=scopeSelect.value || "home";

  let html="<option value='home'>집 전체</option>";
  for(let i=1;i<=count;i++){
    html += "<option value='"+i+"'>"+i+"영역</option>";
  }
  scopeSelect.innerHTML=html;
  const values=Array.from(scopeSelect.options).map(o=>o.value);
  scopeSelect.value=values.includes(before)?before:"home";
}


function renderPlan(){
  if(!$('planSummary'))return;
  refreshScopeSelect();
  const conditionPanel=$('conditionPanel');
  const predictBtn=$('predictBtn');
  const conditionTitle=$('conditionTitle');
  const learnBtn=$('learnBtn');
  const learnActions=$('learnActions');
  const cleanExecuteBtn=$('cleanExecuteBtn');
  const learnPill=$('learnPill');
  const learnStatus=$('learnStatus');
  const learnFill=$('learnFill');
  const learnSteps=$('learnSteps');
  const learnTitle=$('learnTitle');
  const learnDesc=$('learnDesc');
  const firstLearnInputs=$('firstLearnInputs');
  const predictionInputs=$('predictionInputs');
  const startCleanPrimary=$('startCleanPrimary');
  const flowGuide=$('flowGuide');
  const mapSectionContent=$('mapSectionContent');
  const mapSectionBadge=$('mapSectionBadge');
  if(flowGuide){
    const guideText=guideForCurrentState();
    let tone=state.userGuideTone||"normal";
    if(state.soc<15)tone="danger";
    else if(state.charging)tone="charging";
    else if(state.predicted && state.soc<state.targetSoc)tone="warning";
    else if(state.missionDone || state.celebrating)tone="done";
    flowGuide.className="flow-guide"+(tone&&tone!=="normal"?" "+tone:"");
    flowGuide.innerHTML="<span class='guide-step'>다음 안내</span>"+guideText;
  }

  if(mapSectionContent){
    if(state.profileReady && !state.mapping){
      setHtml(mapSectionContent,getLearnedMapHtml());
      if(mapSectionBadge){
        mapSectionBadge.textContent=getHomeSizeLabel(state.areaPyung)+' · '+getDisplayZoneCount()+'개 영역';
        mapSectionBadge.className='home-section-badge ready';
      }
    }else if(state.mapping){
      setHtml(mapSectionContent,"<div class='map-empty-state learning'><span class='map-empty-icon'>🧭</span><b>우리 집 맵을 학습 중이에요</b><span>집 구조와 바닥 상태를 기록하고 있어요.<br>학습 진행 "+state.mappingProgress+"%</span></div>");
      if(mapSectionBadge){
        mapSectionBadge.textContent='학습 '+state.mappingProgress+'%';
        mapSectionBadge.className='home-section-badge learning';
      }
    }else{
      setHtml(mapSectionContent,"<div class='map-empty-state'><span class='map-empty-icon'>🏠</span><b>우리 집 맵을 준비하고 있어요</b><span>아래 맞춤 청소 준비에서 1회차 학습을 시작하면<br>방 구조와 바닥 상태가 여기에 표시돼요.</span></div>");
      if(mapSectionBadge){
        mapSectionBadge.textContent='학습 전';
        mapSectionBadge.className='home-section-badge';
      }
    }
  }

  if(learnSteps){
    learnSteps.classList.remove('map-ready');
    learnSteps.innerHTML=mappingSteps.map((s,i)=>{
      let cls='learn-step';
      if(state.profileReady || i<state.mappingStepIndex)cls+=' done';
      else if(state.mapping && i===state.mappingStepIndex)cls+=' active';
      return '<div class="'+cls+'">'+s.label+'</div>';
    }).join('');
  }
  if(learnFill)learnFill.style.width=(state.profileReady?100:state.mappingProgress)+'%';

  if(state.mapping){
    if(learnTitle)learnTitle.textContent="로보킹이 우리 집을 배우고 있어요";
    if(learnDesc)learnDesc.textContent="맵·바닥 상태·배터리 사용량을 차례로 기록해요.";
    if(learnPill)learnPill.textContent="학습 중";
    if(learnStatus){
      const currentStep=mappingSteps[state.mappingStepIndex]||mappingSteps[0];
      learnStatus.innerHTML=currentStep.label+" 중 · "+state.mappingProgress+"%<br><b>배터리 "+Math.round(state.firstRunStartSoc)+"% → "+Math.round(state.soc)+"%</b>";
    }
    if(learnBtn){learnBtn.textContent="로보킹이 집을 배우는 중...";learnBtn.disabled=true;}
    if(conditionPanel)conditionPanel.classList.add('locked-area');
  }else if(state.profileReady){
    if(learnTitle)learnTitle.textContent="우리 집 정보로 맞춤 청소를 준비해요";
    if(learnDesc)learnDesc.textContent="저장된 맵과 바닥 상태를 이용해 필요한 청소만 준비해요.";
    if(learnPill)learnPill.textContent="프로필 저장됨";
    if(learnStatus)learnStatus.innerHTML="매핑 완료 · "+getHomeSizeLabel(activeRun.areaPyung)+" 집 구조 저장";
    if(learnBtn){learnBtn.textContent="🔄 학습 다시 실행";learnBtn.disabled=false;learnBtn.classList.add('ready');}
    if(conditionPanel)conditionPanel.classList.remove('locked-area');
  }else{
    if(learnTitle)learnTitle.textContent="처음 사용할 때는 로보킹이 집을 먼저 배워요";
    if(learnDesc)learnDesc.textContent="처음 한 번만 집 구조와 바닥 상태를 배워요.";
    if(learnPill)learnPill.textContent="초기 학습";
    if(learnStatus)learnStatus.textContent="시작 버튼을 누르면 집 정보를 저장해요.";
    if(learnBtn){learnBtn.textContent="🏠 1회차 학습 청소 시작";learnBtn.disabled=false;learnBtn.classList.remove('ready');}
    if(conditionPanel)conditionPanel.classList.remove('locked-area');
  }

  if(firstLearnInputs)firstLearnInputs.style.display=state.profileReady?'none':'block';
  if(predictionInputs)predictionInputs.style.display=state.profileReady?'block':'none';
  if(learnActions){
    learnActions.classList.toggle('ready',state.profileReady && !state.mapping);
  }
  if(cleanExecuteBtn){
    const showCleanExecute=state.profileReady && !state.mapping;
    cleanExecuteBtn.style.display=showCleanExecute?'block':'none';
    cleanExecuteBtn.disabled=!showCleanExecute || state.cleaning || state.charging || state.mapping || state.predicting;

    if(state.cleaning){
      cleanExecuteBtn.textContent='🧹 청소 중...';
    }else if(state.charging){
      cleanExecuteBtn.textContent='🔋 충전 중...';
    }else if(!state.predicted){
      cleanExecuteBtn.textContent='🧹 청소하기';
    }else if(state.soc<state.targetSoc){
      cleanExecuteBtn.textContent='🔋 충전 후 청소';
    }else{
      cleanExecuteBtn.textContent='🧹 바로 청소';
    }
  }


  if(predictBtn){
    const mainBtnDisabled=!state.profileReady || state.mapping || state.predicting || state.cleaning || state.charging;
    const key=getManualSelectionKey();
    const manualReady=state.profileReady && state.predicted && state.smartCleanMode==="manual" && state.manualReady && state.manualKey===key;

    predictBtn.disabled=mainBtnDisabled;
    predictBtn.style.opacity=mainBtnDisabled?'.55':'1';
    predictBtn.classList.toggle('running',state.predicting||state.charging||state.cleaning);

    if(!state.profileReady){
      predictBtn.textContent='🤖 학습 후 사용 가능';
    }else if(state.predicting){
      predictBtn.textContent='🤖 준비 중...';
    }else if(state.charging){
      predictBtn.textContent='🔋 충전 중...';
    }else if(state.cleaning){
      predictBtn.textContent='🧹 청소 중...';
    }else if(manualReady && state.soc<state.targetSoc){
      predictBtn.textContent='🔋 선택 조건으로 충전하고 청소하기';
    }else if(manualReady){
      predictBtn.textContent='🧹 선택 조건으로 바로 청소하기';
    }else{
      predictBtn.textContent='🔥 선택 조건으로 준비하고 청소하기';
    }
  }
  if(conditionPanel){
    conditionPanel.classList.toggle('manual-mode',state.profileReady);
  }
  if(conditionTitle){
    conditionTitle.textContent='✍️ 직접 조건 청소';
  }

  document.querySelectorAll('.scope-btn').forEach(btn=>{
    btn.classList.remove('active');
    btn.disabled=!state.profileReady || !state.predicted || state.mapping;
    btn.style.opacity=(!state.profileReady || !state.predicted || state.mapping)?'.55':'1';
  });
  if(state.selectedScope==="home")$('scopeHome').classList.add('active');
  else if($('scopeZone'+state.selectedZone))$('scopeZone'+state.selectedZone).classList.add('active');

  $('planModel').textContent=state.mapping?'집 배우는 중':(!state.profileReady?'처음 학습 전':(state.predicted?'로보킹 맞춤 준비':'우리 집 저장 완료'));


  if(startCleanPrimary){
    const canStart=state.profileReady && state.predicted && !state.mapping && !state.cleaning && !state.charging;
    startCleanPrimary.disabled=!canStart;
    if(state.mapping){
      startCleanPrimary.innerHTML='🏠 우리 집을 배우는 중이에요<small id="startCleanHint">학습이 끝나면 청소 미션을 시작할 수 있어요</small>';
    }else if(!state.profileReady){
      startCleanPrimary.innerHTML='🏠 1회차 학습 청소가 먼저예요<small id="startCleanHint">집 구조를 저장한 뒤 청소할 수 있어요</small>';
    }else if(!state.predicted){
      startCleanPrimary.innerHTML='🤖 오늘 청소 준비가 먼저예요<small id="startCleanHint">로보킹이 필요한 만큼 알아서 준비해요</small>';
    }else if(state.soc<state.targetSoc){
      startCleanPrimary.innerHTML='🔋 충전하고 청소하기<small id="startCleanHint">필요한 만큼만 채우고 출발해요</small>';
    }else{
      startCleanPrimary.innerHTML='🧹 청소 미션 수행하기<small id="startCleanHint">'+state.selectedLabel+' · 바로 출동 가능</small>';
    }
  }

  if(!state.profileReady){
    $('planTargetSoc').textContent='--';
    $('planSummary').innerHTML="<div class='summary-card'><div class='summary-title'>학습 대기</div><div class='summary-row'><span class='summary-key'>상태</span><span class='summary-val'>집 정보 없음</span></div><div class='summary-row'><span class='summary-key'>다음 단계</span><span class='summary-val green'>1회차 학습 시작</span></div></div>";
    $('planSocSub').textContent="학습 후 표시";
    return;
  }
  if(!state.predicted){
    $('planTargetSoc').textContent='--';
    $('planSummary').innerHTML="<div class='summary-card'><div class='summary-title'>우리 집 저장 완료</div><div class='summary-row'><span class='summary-key'>집 크기</span><span class='summary-val'>"+state.areaPyung+"평 · "+state.cleaningAreaM2+"㎡</span></div><div class='summary-row'><span class='summary-key'>학습 중 사용</span><span class='summary-val em'>"+Math.round(state.firstRunStartSoc)+"% → "+Math.round(state.soc)+"%</span></div><div class='summary-row'><span class='summary-key'>다음 단계</span><span class='summary-val green'>오늘 청소 준비하기</span></div></div>";
    $('planSocSub').textContent="준비 대기";
    return;
  }

  $('planTargetSoc').textContent=state.targetSoc;
  const scopeText=state.selectedScope==="home"?"집 전체":state.selectedLabel;
  const detail=state.selectedScope==="home"
    ? state.areaPyung+"평 프로필 · "+state.cleaningAreaM2+"㎡"
    : (state.floorType||"바닥재질")+" · 오염도 "+(state.dirtLevel||"-")+" · "+state.cleaningAreaM2+"㎡";
  const conditionDetail=state.cleanModeLabel+" · "+state.intensityLabel+" · "+state.todayStateLabel;
  $('planSummary').innerHTML="<div class='summary-card'>"
    +"<div class='summary-title'>"+scopeText+"</div>"
    +"<div class='summary-row'><span class='summary-key'>조건</span><span class='summary-val'>"+conditionDetail+"</span></div>"
    +"<div class='summary-row'><span class='summary-key'>프로필</span><span class='summary-val'>"+detail+"</span></div>"
    +"<div class='summary-row'><span class='summary-key'>청소 준비</span><span class='summary-val em'>완료</span></div>"
    +"<div class='summary-row'><span class='summary-key'>준비 방식</span><span class='summary-val green'>"+state.matchNote+"</span></div>"
    +"</div>";
  $('planSocSub').textContent="필요한 만큼만 충전";
}

function renderHome(){
  const room=$("room");room.className="room";
  if(!state.cleaning && state.robotMotion==="returning")room.classList.add("returning");
  if(!state.cleaning && state.robotMotion==="docked")room.classList.add("docked");
  if(!state.cleaning && state.robotMotion==="departing")room.classList.add("departing");
  if(state.predicted && !state.cleaning && !state.charging && !state.chargeComplete)room.classList.add("route-preview");

  if(state.chargeComplete){
    room.classList.add("celebrate");
    if(state.chargePurpose==="nextHome"){
      $("speech").innerHTML="<strong style='color:#2f8b3a'>다음 청소 준비 완료!</strong><br>필요한 만큼 채워뒀어요.";
      setModeChipText("✅ 다음 전체 청소 준비 완료");
      $("batteryFace").textContent="😊";$("spark").textContent="💖";
      $("batteryMessage").innerHTML="다음 집 전체 청소까지<br>미리 준비해뒀어요.";
      $("timeTip").textContent="다음 청소도 바로 시작할 수 있어요.";
    }else{
      $("speech").innerHTML="<strong>배불러요!</strong><br>이제 청소 가능해요!";
      setModeChipText("💖 충전 완료 · 출동 준비");
      $("batteryFace").textContent="😍";$("spark").textContent="💖";
      $("batteryMessage").innerHTML="필요한 만큼 채웠어요.<br>출동 준비 완료!";
      $("timeTip").textContent=state.selectedLabel+" 청소를 시작할 수 있어요.";
    }
  }else if(state.celebrating){
    room.classList.add("celebrate");
    $("speech").innerHTML="<strong>청소 완료!</strong><br>보상을 받았어요!";
    setModeChipText("🏆 미션 완료 · +50 코인");
    $("batteryFace").textContent="🥳";$("spark").textContent="🎉";
  }else if(state.mapping){
    room.classList.add("cleaning");
    const step=mappingSteps[state.mappingStepIndex]||mappingSteps[0];
    $("speech").innerHTML="<strong style='color:#2f8b3a'>우리 집을 배우는 중!</strong><br>"+step.label+" 중이에요.";
    setModeChipText("🏠 학습 청소 · 배터리 "+Math.round(state.firstRunStartSoc)+"% → "+Math.round(state.soc)+"%");
    $("batteryFace").textContent="🧭";$("spark").textContent="📡";
    $("batteryMessage").innerHTML="학습 청소 중입니다.<br>배터리가 실제로 소모돼요.";
    $("timeTip").textContent="학습 진행 "+state.mappingProgress+"% · 현재 배터리 "+Math.round(state.soc)+"%";
  }else if(state.predicting){
    $("speech").innerHTML="<strong style='color:#2f8b3a'>준비 중이에요!</strong><br>오늘 상태에 맞춰 준비하고 있어요.";
    setModeChipText("🤖 우리 집 기록으로 준비 중");
    $("batteryFace").textContent="🤔";$("spark").textContent="✨";
  }else if(!state.profileReady){
    $("speech").innerHTML="<strong style='color:#ef8c32'>처음 만났어요!</strong><br>1회차 청소로 우리 집을 알려주세요.";
    setModeChipText("🏠 집 구조 학습 필요");
    $("batteryFace").textContent="🙂";$("spark").textContent="✨";
    $("batteryMessage").innerHTML="아직 우리 집 정보를 몰라요.<br>학습 청소가 필요합니다.";
    $("timeTip").textContent="1회차 학습 후 청소 준비 가능";
  }else if(!state.predicted){
    $("speech").innerHTML="<strong>집을 배웠어요!</strong><br>이제 청소 준비를 맡겨주세요.";
    setModeChipText("✅ 우리 집 저장 완료");
    $("batteryFace").textContent="😊";$("spark").textContent="✨";
    $("batteryMessage").innerHTML="집 구조 학습 완료!<br>오늘 청소 준비하기를 눌러주세요.";
    $("timeTip").textContent="청소 준비 대기 중";
  }else if(state.cleaning){
    room.classList.add("cleaning");
    $("speech").innerHTML="<strong>열심히 청소 중이에요!</strong><br>진행률 "+state.progress+"%";
    setModeChipText("🧹 "+state.selectedLabel+" 청소 중 · "+state.progress+"%");
    $("batteryFace").textContent="🧹";
    $("batteryMessage").innerHTML="청소 중입니다.<br>로보킹이 청소하면서 배터리를 사용하고 있어요.";
    $("timeTip").textContent="청소 진행률 "+state.progress+"%";
    $("spark").textContent="💨";
  }else if(state.charging){
    room.classList.add("charging");
    if(state.robotMotion==="returning"){
      if(state.chargePurpose==="nextHome"){
        $("speech").innerHTML="<strong style='color:#e48627'>스테이션으로 돌아가요</strong><br>다음 청소를 미리 준비할게요.";
        setModeChipText("🏠 다음 청소 준비 중");
      }else{
        $("speech").innerHTML="<strong style='color:#e48627'>스테이션으로 가는 중!</strong><br>잠깐 힘을 채우고 올게요.";
        setModeChipText("🏠 충전 스테이션 복귀 중");
      }
    }else{
      if(state.chargePurpose==="nextHome"){
        $("speech").innerHTML="<strong style='color:#e48627'>다음 청소 준비 중!</strong><br>필요한 만큼만 미리 채울게요.";
        setModeChipText("⚡ 다음 전체 청소 준비 중");
      }else{
        $("speech").innerHTML="<strong style='color:#e48627'>잠깐 쉬는 중이에요</strong><br>필요한 만큼만 충전할게요.";
        setModeChipText("⚡ "+state.selectedLabel+" 출동 준비 중");
      }
    }
    $("batteryFace").textContent="😌";
    $("batteryMessage").innerHTML="충전 스테이션에서 쉬면서<br>필요한 만큼만 채우고 있어요.";
    $("timeTip").textContent="로보킹이 필요한 만큼만 채우고 있어요.";
    $("spark").textContent="⚡";
  }else if(state.soc<15){
    room.classList.add("low");
    $("speech").innerHTML="<strong style='color:#ef4e45'>배가 너무 고파요...</strong><br>충전이 필요해요.";
    setModeChipText("⚠️ 배터리 부족");
    $("batteryFace").textContent="🥴";
    $("batteryMessage").innerHTML="배터리가 부족해요.<br>먼저 충전해 주세요.";
    $("timeTip").textContent="충전 후 청소를 시작해 주세요.";
    $("spark").textContent="💦";
  }else{
    setModeChipText("✨ 로보킹 맞춤 준비");
    $("batteryFace").textContent=state.soc>90?"😮":"😊";
    $("spark").textContent="✨";

    if(state.soc < state.targetSoc){
      $("speech").innerHTML=
        "<strong style='color:#ef8c32'>아직 배고파요!</strong><br>"
        + "필요한 만큼만<br>충전하고 청소할게요.";
      $("batteryMessage").innerHTML=state.selectedLabel+" 청소를 위해<br>조금 더 충전이 필요해요.";
      $("timeTip").textContent="잠깐 충전하면 청소를 시작할 수 있어요.";
    }else{
      $("speech").innerHTML="<strong>배가 든든해요!</strong><br>"+state.selectedLabel+" 청소를 준비할게요!";
      $("batteryMessage").innerHTML="현재 배터리로 충분해요.<br>바로 출동할 수 있어요.";
      $("timeTip").textContent="현재 배터리로 "+state.selectedLabel+" 청소가 가능합니다.";
    }
  }
}

/* ============================================================
   PAGE 2 · 부품 케어 (부품 상태 카드 + 수명 요약 + 실시간 케어 기록)
   ============================================================ */
function totalCleanCount(){return DEMO_CLEAN_BASE+Number(state.cleanCount||0);}
function getPartStatuses(){
  const total=totalCleanCount();
  const wheels = total>=60 ? {level:"bad",text:"점검이 필요해요"} : (total>=30 ? {level:"check",text:"한번 살펴보면 좋아요"} : {level:"good",text:"평소와 비슷해요"});
  const brush  = total>=40 ? {level:"bad",text:"머리카락을 제거해 주세요"} : (total>=8 ? {level:"check",text:"한번 살펴보면 좋아요"} : {level:"good",text:"깨끗해요"});
  const filter = total>=50 ? {level:"bad",text:"교체 시기가 됐어요"} : (total>=25 ? {level:"check",text:"한번 살펴보면 좋아요"} : {level:"good",text:"괜찮아요"});
  let battery={level:"good",text:"편안해요"};
  if(state.charging)battery={level:"good",text:"쉬면서 힘을 채우고 있어요"};
  else if(state.soc<15)battery={level:"bad",text:"배가 고파요"};
  else if(state.temperature>34)battery={level:"check",text:"조금 더워요, 쉬어갈게요"};
  const faces={good:"🙂",check:"😐",bad:"😟"};
  return [
    {key:"wheel",icon:"🛞",name:"바퀴 상태",...wheels,face:faces[wheels.level],
      detail:"누적 청소 "+total+"회 · 바퀴 마모가 적어 평소처럼 부드럽게 달릴 수 있어요.",tip:"바퀴 틈에 낀 실이나 머리카락은 한 달에 한 번만 빼주면 충분해요."},
    {key:"brush",icon:"🧹",name:"브러시 상태",...brush,face:faces[brush.level],
      detail:"누적 청소 "+total+"회 · 브러시에 머리카락과 털이 조금 감겼을 수 있어요.",tip:"브러시를 빼서 감긴 털을 잘라내면 흡입력이 돌아오고 모터 부담도 줄어요.",coupon:true},
    {key:"filter",icon:"🧊",name:"필터 상태",...filter,face:faces[filter.level],
      detail:"누적 청소 "+total+"회 · 필터 막힘이 적어 흡입 효율이 좋아요.",tip:"필터는 2~3주마다 톡톡 털어주고, 6개월마다 교체하면 좋아요.",coupon:true},
    {key:"battery",icon:"🔋",name:"배터리 컨디션",...battery,face:faces[battery.level],
      detail:"현재 "+state.soc+"% · 온도 "+state.temperature+"℃ · 건강도 "+state.health+"%",tip:"완충 대신 필요한 만큼만 채우고, 15%를 남겨 쉬어가면 배터리 수명이 오래가요."}
  ];
}
function renderCare(){
  const grid=$("partsGrid");
  if(!grid)return;
  const parts=getPartStatuses();
  setHtml(grid,parts.map(p=>
    "<button type='button' class='part-card "+p.level+"' data-action='partDetail' data-part='"+p.key+"'>"
    +"<div class='part-icon'>"+p.icon+"</div>"
    +"<div class='part-info'><div class='part-name'>"+p.name+"</div><div class='part-status'>"+p.text+"</div></div>"
    +"<div class='part-face'>"+p.face+"</div></button>").join(""));
  const a=$("careAcceptText"); if(a)a.textContent=state.acceptCount;
  const r=$("careReserveText"); if(r)r.textContent=state.reserveGuardCount;
  const s=$("careSavedText"); if(s)s.textContent=Math.round(state.savedChargePct);
  const hf=$("careHealthFill"); if(hf)hf.style.width=clamp(state.health,0,100)+"%";
  const ht=$("careHealthText"); if(ht)ht.textContent=clamp(Math.round(state.health),0,100)+"%";
  const note=$("careNote");
  if(note){
    if(state.charging)note.textContent="지금 "+state.targetSoc+"%까지만 채우고 있어요. 완충보다 "+(100-state.targetSoc)+"% 덜 채워 배터리 부담을 줄여요.";
    else if(state.cleaning)note.textContent="청소 중이에요. 15%가 되면 무리하지 않고 스스로 쉬어가요.";
    else if(state.mapping)note.textContent="학습 청소 중에도 배터리 15% 이상은 항상 남겨두고 있어요.";
    else note.textContent="완충 대신 필요한 만큼만 채운 덕분에 지금까지 충전량 "+Math.round(state.savedChargePct)+"%를 덜 채웠어요. 오늘도 과충전 없이 관리 중이에요.";
  }
}
function openPartDetail(el){
  const key=el && el.dataset ? el.dataset.part : el;
  const p=getPartStatuses().find(x=>x.key===key);
  if(!p)return;
  const body="<div class='modal-emoji'>"+p.icon+" "+p.face+"</div><b>"+p.text+"</b><br><br>"+p.detail+"<br><br>💡 "+p.tip;
  if(p.coupon){
    openModal(p.name,body,{showCancel:true,cancelText:"닫기",confirmText:"소모품 쿠폰 보기",onConfirm:()=>{closeModal();state.rewardTab="coupons";switchPage("rewardPage");showToast("코인으로 클린 키트 쿠폰을 교환할 수 있어요.");}});
  }else{
    openModal(p.name,body);
  }
}

/* ============================================================
   PAGE 3 · 예약 청소 (출퇴근 맞춤 예약 + 테마 기간 청소)
   ============================================================ */
const dayNames=["일","월","화","수","목","금","토"];
const themeDefs=[
  {key:"chuseok",icon:"🌕",name:"추석 맞이 대청소",start:"2026-09-18",end:"2026-09-24",desc:"가족과 손님이 오기 전, 현관과 거실을 매일 한 번씩 깨끗하게 준비해요.",chips:["집 전체","건식+물걸레","매일 1회","현관·거실 집중"]},
  {key:"season",icon:"🍂",name:"환절기 먼지 케어",start:"2026-09-01",end:"2026-10-15",desc:"창문을 자주 여는 시기라 침실 먼지를 꼼꼼 모드로 관리해요.",chips:["침실 중심","꼼꼼","주 3회","필터 점검 알림"]},
  {key:"yearend",icon:"🎄",name:"연말 대청소",start:"2026-12-20",end:"2026-12-31",desc:"한 해를 마무리하며 구역별로 나눠 무리 없이 집 전체를 정리해요.",chips:["구역 나눔","물걸레","격일","배터리 분할"]},
  {key:"spring",icon:"🌸",name:"봄맞이 새단장",start:"2027-03-01",end:"2027-03-31",desc:"꽃가루와 미세먼지가 많은 봄, 현관 매트와 거실 러그를 집중 관리해요.",chips:["현관·거실","건식","주 4회","러그 집중"]},
  {key:"rainy",icon:"☔",name:"장마철 물걸레 케어",start:"2026-06-20",end:"2026-07-20",desc:"습한 바닥을 물걸레로 자주 닦아 끈적임과 냄새를 줄여요.",chips:["마루·타일","물걸레","매일 1회","건조 시간 확보"]},
  {key:"movein",icon:"🏠",name:"이사·입주 집중 청소",start:null,end:null,desc:"새집 첫 3일, 먼지가 많은 구역부터 순서대로 집중 청소해요.",chips:["3일 집중","더러운 곳 우선","하루 2회"]}
];
function parseDay(s){return s?new Date(s+"T00:00:00"):null;}
function fmtMD(s){const d=parseDay(s);return d?(d.getMonth()+1)+"월 "+d.getDate()+"일":"";}
function themeStatus(t){
  if(!t.start)return {kind:"any",label:"언제든"};
  const today=new Date();today.setHours(0,0,0,0);
  const s=parseDay(t.start), e=parseDay(t.end);
  if(today>e)return {kind:"past",label:"지난 시즌"};
  if(today<s){const dday=Math.ceil((s-today)/86400000);return {kind:"soon",label:"D-"+dday};}
  return {kind:"live",label:"진행 중"};
}
const toMin=(hhmm)=>{const [h,m]=String(hhmm||"09:00").split(":").map(Number);return h*60+(m||0);};
const fmtMin=(m)=>{m=((m%1440)+1440)%1440;return String(Math.floor(m/60)).padStart(2,"0")+":"+String(m%60).padStart(2,"0");};
function estimateCleanMinutes(){
  const req=Number(state.profileReady?(state.predicted?state.requiredSoc:activeRun.home.requiredSoc):0);
  return req>0?Math.max(15,Math.round(req*1.4)):40;
}
function commutePlan(){
  const leave=$("leaveTime")?$("leaveTime").value:"09:00";
  const ret=$("returnTime")?$("returnTime").value:"18:30";
  const mins=estimateCleanMinutes();
  let start,end;
  if(state.commuteMode==="before"){end=toMin(ret)-60;start=end-mins;}
  else{start=toMin(leave)+30;end=start+mins;}
  return {leave,ret,mins,start:fmtMin(start),end:fmtMin(end)};
}
function commuteDayText(){
  const d=(state.commuteDays||[]).slice().sort((a,b)=>a-b);
  if(d.length===7)return "매일";
  if(d.join()==="1,2,3,4,5")return "평일";
  if(d.join()==="0,6")return "주말";
  return d.map(i=>dayNames[i]).join("·");
}
function nextCommuteRun(){
  if(!state.commuteOn||!(state.commuteDays||[]).length)return null;
  const plan=commutePlan();const now=new Date();
  for(let add=0;add<8;add++){
    const d=new Date(now);d.setDate(now.getDate()+add);
    if(!state.commuteDays.includes(d.getDay()))continue;
    const [h,m]=plan.start.split(":").map(Number);
    const startAt=new Date(d);startAt.setHours(h,m,0,0);
    if(startAt<=now)continue;
    const label=add===0?"오늘":(add===1?"내일":(d.getMonth()+1)+"월 "+d.getDate()+"일("+dayNames[d.getDay()]+")");
    return {label,plan};
  }
  return null;
}
function renderSchedule(){
  const sw=$("commuteSwitch");
  if(!sw)return;
  sw.classList.toggle("on",state.commuteOn);
  const body=$("commuteBody"); if(body)body.classList.toggle("off",!state.commuteOn);
  const chips=$("dayChips");
  if(chips)setHtml(chips,dayNames.map((n,i)=>"<button type='button' class='day-chip"+(state.commuteDays.includes(i)?" on":"")+"' data-action='toggleDay' data-day='"+i+"'>"+n+"</button>").join(""));
  const after=$("commuteAfter"),before=$("commuteBefore");
  if(after)after.classList.toggle("active",state.commuteMode==="after");
  if(before)before.classList.toggle("active",state.commuteMode==="before");

  const prev=$("commutePreview");
  if(prev){
    prev.classList.toggle("off",!state.commuteOn);
    if(!state.commuteOn){prev.textContent="스위치를 켜면 출퇴근 시간에 맞춘 예약이 만들어져요.";}
    else{
      const plan=commutePlan();
      const target=state.profileReady?(state.predicted?state.targetSoc:activeRun.home.targetSoc):null;
      prev.innerHTML="<b>"+commuteDayText()+"</b> "+plan.start+" 출발 → 약 <b>"+plan.mins+"분</b> 청소 후 "+plan.end+" 도킹"
        +(target?"<br>준비 배터리 <b>"+target+"%</b>만 채우고 출발해요.":"<br>1회차 학습을 마치면 준비 배터리와 시간이 우리 집에 맞게 정해져요.")
        +"<br>퇴근("+plan.ret+") 전에는 항상 조용히 마무리해요.";
    }
  }

  const list=$("themeList");
  if(list)setHtml(list,themeDefs.map(t=>{
    const st=themeStatus(t);const on=!!state.activeThemes[t.key];
    const period=t.start?fmtMD(t.start)+" ~ "+fmtMD(t.end):"원하는 날부터 3일";
    return "<div class='theme-card"+(on?" on":"")+(st.kind==="past"?" past":"")+"'>"
      +"<div class='theme-icon'>"+t.icon+"</div>"
      +"<div class='theme-info'><div class='theme-name'>"+t.name+"<span class='theme-state "+st.kind+"'>"+st.label+"</span></div>"
      +"<div class='theme-period'>"+period+"</div><div class='theme-desc'>"+t.desc+"</div>"
      +"<div class='theme-chips'>"+t.chips.map(c=>"<span>"+c+"</span>").join("")+"</div></div>"
      +"<button type='button' class='theme-btn"+(on?" on":"")+"' data-action='toggleTheme' data-theme='"+t.key+"'>"+(on?"예약됨 ✓":"예약하기")+"</button>"
      +"</div>";
  }).join(""));

  const up=$("upcomingList");
  if(up){
    const items=[];
    const next=nextCommuteRun();
    if(next)items.push({icon:"🚶",title:next.label+" "+next.plan.start+" 출퇴근 맞춤 청소",desc:commuteDayText()+" 반복 · "+next.plan.end+" 전 도킹 완료"});
    themeDefs.forEach(t=>{
      if(!state.activeThemes[t.key])return;
      const st=themeStatus(t);
      let desc;
      if(st.kind==="live")desc="진행 중 · "+fmtMD(t.end)+"까지 · "+t.chips[2];
      else if(st.kind==="soon")desc=fmtMD(t.start)+"부터 시작 ("+st.label+") · "+t.chips[2];
      else if(st.kind==="past")desc="이번 시즌은 지났어요. 내년 같은 시기에 다시 알려드려요.";
      else desc="시작 날짜를 정하면 3일 동안 집중 청소해요.";
      items.push({icon:t.icon,title:t.name,desc});
    });
    const badge=$("upcomingBadge"); if(badge)badge.textContent=items.length+"건";
    setHtml(up,items.length?items.map(i=>"<div class='upcoming-item'><div class='upcoming-icon'>"+i.icon+"</div><div><strong>"+i.title+"</strong><span>"+i.desc+"</span></div></div>").join("")
      :"<div class='upcoming-empty'>아직 예약이 없어요. 출퇴근 예약을 켜거나 테마 청소를 예약해 보세요.</div>");
  }
}
function fillTimeSelects(){
  const leave=$("leaveTime"),ret=$("returnTime");
  if(leave&&!leave.options.length){for(let m=6*60;m<=11*60;m+=30){const o=document.createElement("option");o.value=fmtMin(m);o.textContent=fmtMin(m);leave.appendChild(o);}leave.value="09:00";}
  if(ret&&!ret.options.length){for(let m=15*60;m<=22*60;m+=30){const o=document.createElement("option");o.value=fmtMin(m);o.textContent=fmtMin(m);ret.appendChild(o);}ret.value="18:30";}
}
function toggleCommute(){
  state.commuteOn=!state.commuteOn;
  render();
  if(state.commuteOn){
    const p=commutePlan();
    addEvent("출퇴근 예약 설정",commuteDayText()+" "+p.start+" 출발, "+p.end+" 도킹으로 예약했어요. 필요한 만큼만 충전한 뒤 출발해요.","예약");
    showToast("출퇴근 맞춤 예약을 켰어요. 집을 비운 시간에만 청소해요.");
  }else showToast("출퇴근 맞춤 예약을 껐어요.");
}
function toggleDay(el){
  const d=Number(el.dataset.day);
  const idx=state.commuteDays.indexOf(d);
  if(idx>=0)state.commuteDays.splice(idx,1);else state.commuteDays.push(d);
  render();
}
function setCommuteMode(el){state.commuteMode=el.dataset.mode||"after";render();}
function toggleTheme(el){
  const key=el.dataset.theme;const t=themeDefs.find(x=>x.key===key);if(!t)return;
  state.activeThemes[key]=!state.activeThemes[key];
  render();
  if(state.activeThemes[key]){
    const st=themeStatus(t);
    addEvent("테마 청소 예약",t.name+"을(를) 예약했어요."+(t.start?" ("+fmtMD(t.start)+" ~ "+fmtMD(t.end)+")":""),"예약");
    showToast(t.icon+" "+t.name+" 예약 완료! "+(st.kind==="live"?"오늘부터 진행해요.":st.kind==="soon"?fmtMD(t.start)+"부터 시작해요.":"시작 시기에 알려드려요."));
  }else showToast(t.name+" 예약을 취소했어요.");
}

/* ============================================================
   PAGE 4 · 이벤트 (오늘의 발견 / 미션 / 사진첩)
   ============================================================ */
const defaultLostItems=[
  {id:"l1",emoji:"🎀",title:"거실 소파 옆",desc:"작은 머리끈으로 보여요.",place:"거실",spot:"소파 옆",time:"오늘 오후 2:15",found:false},
  {id:"l2",emoji:"🔑",title:"현관 매트 근처",desc:"작은 열쇠로 보여요.",place:"현관",spot:"매트 근처",time:"오늘 오전 10:08",found:false},
  {id:"l3",emoji:"🧦",title:"침대 옆 바닥",desc:"양말 한 짝으로 보여요.",place:"침실",spot:"침대 옆",time:"어제 오후 8:42",found:false}
];
const lostImages=(mediaData&&mediaData.lostItems)||[];
const lostItems=defaultLostItems.map((it,i)=>{
  const img=lostImages[i];const o=Object.assign({},it);
  if(img){o.src=img.src;if(img.place)o.place=img.place;if(img.time)o.time=img.time;if(img.note)o.desc=img.note;if(img.title&&img.title!==img.name.replace(/\.[^.]+$/,""))o.title=img.title;}
  return o;
});
lostImages.slice(defaultLostItems.length).forEach((img,i)=>{
  lostItems.push({id:"lx"+i,emoji:"📦",src:img.src,title:img.title||"청소 중 발견",desc:img.note||"청소 중 바닥에서 발견했어요.",place:img.place||"거실",spot:"",time:img.time||"오늘",found:false});
});
const demoPhotos=[
  {emoji:"🐱",title:"소파 위에서 낮잠",place:"거실",time:"오늘 오후 1:20",note:"햇살 아래에서 곤히 자고 있어요."},
  {emoji:"🐶",title:"창밖 구경 중",place:"거실",time:"오늘 오전 11:05",note:"밖에 지나가는 새를 한참 봤어요."},
  {emoji:"🐾",title:"현관 앞 기다리기",place:"현관",time:"어제 오후 6:40",note:"퇴근 시간이 가까워지면 여기서 기다려요."},
  {emoji:"🐈",title:"침대 밑 탐험",place:"침실",time:"어제 오후 3:12",note:"로보킹과 마주쳐서 잠깐 놀랐어요."}
];
const realPhotos=(mediaData&&mediaData.photos)||[];
const photos=realPhotos.length?realPhotos:demoPhotos;
const usingDemoPhotos=!realPhotos.length;

const foundMapSpots={"거실":[122,58],"현관":[179,112],"침실":[34,40],"주방":[34,112],"다용도":[179,40]};
function foundMapSvg(item){
  const spot=foundMapSpots[item.place]||foundMapSpots["거실"];
  const px=spot[0],py=spot[1];
  return "<svg viewBox='0 0 200 150' role='img' aria-label='발견 위치 지도'>"
    +"<rect class='fm-room' x='5' y='5' width='58' height='68' rx='8'/><text class='fm-label' x='34' y='44'>침실</text>"
    +"<rect class='fm-room' x='5' y='78' width='58' height='67' rx='8'/><text class='fm-label' x='34' y='116'>주방</text>"
    +"<rect class='fm-room' x='68' y='5' width='90' height='140' rx='8'/><text class='fm-label' x='113' y='130'>거실</text>"
    +"<rect x='80' y='28' width='62' height='26' rx='9' fill='#c9ad82' opacity='.75'/><rect x='88' y='72' width='46' height='20' rx='5' fill='#e5cfa8'/>"
    +"<rect class='fm-room' x='163' y='5' width='32' height='68' rx='8'/><text class='fm-label' x='179' y='44' style='font-size:9px'>다용도</text>"
    +"<rect class='fm-room' x='163' y='78' width='32' height='67' rx='8'/><text class='fm-label' x='179' y='116'>현관</text>"
    +"<g class='fm-pin'><path d='M"+px+" "+(py+14)+" C"+(px-13)+" "+(py-2)+", "+(px-13)+" "+(py-14)+", "+px+" "+(py-14)+" C"+(px+13)+" "+(py-14)+", "+(px+13)+" "+(py-2)+", "+px+" "+(py+14)+" Z' fill='#ef8c32' stroke='#fff' stroke-width='2'/>"
    +"<circle cx='"+px+"' cy='"+(py-5)+"' r='7' fill='#fff'/><text x='"+px+"' y='"+(py-1.5)+"' text-anchor='middle' style='font-size:9px'>🤖</text></g></svg>";
}
function renderFound(){
  const today=lostItems[0];
  const card=$("foundTodayCard");
  if(card&&today){
    card.innerHTML="<div class='found-card-title'>🗓️ 오늘의 발견물</div>"
      +"<div class='found-photo'>"+(today.src?"<img src='"+today.src+"' alt=''>":today.emoji)+"</div>"
      +"<div class='found-name"+(today.found?" done":"")+"'>"+(today.found?"확인 완료 ✅":"작은 물건 발견")+"</div>"
      +"<div class='found-desc'>"+esc(today.desc)+"</div>"
      +"<div class='found-meta'>📍 "+esc((today.place+" "+today.spot).trim())+"<br>🕒 "+esc(today.time)+"</div>"
      +"<div class='found-btn'><button type='button' data-action='foundItem' data-id='"+today.id+"'>더 자세히 보기 ›</button></div>";
  }
  const map=$("foundMap"); if(map&&today)map.innerHTML=foundMapSvg(today);
  const list=$("foundList");
  if(list)list.innerHTML=lostItems.map(it=>"<div class='found-item"+(it.found?" done":"")+"' data-action='foundItem' data-id='"+it.id+"'>"
    +"<div class='found-thumb'>"+(it.src?"<img src='"+it.src+"' alt=''>":it.emoji)+"</div>"
    +"<div><strong>"+esc(it.title)+"</strong><span>"+esc(it.desc)+"</span></div>"
    +"<div class='found-right'>📍 "+esc(it.place)+"<br>🕒 "+esc(it.time)+"</div></div>").join("");
  const badge=$("foundCountBadge"); if(badge)badge.textContent="전체 "+lostItems.length+"건";
}
function openFoundItem(el){
  const id=el&&el.dataset?el.dataset.id:el;
  const it=lostItems.find(x=>x.id===id);if(!it)return;
  const visual=it.src?"<img class='modal-img' src='"+it.src+"' alt=''>":"<div class='modal-emoji'>"+it.emoji+"</div>";
  const body=visual+"<b>"+esc(it.desc)+"</b><br>📍 "+esc((it.place+" "+it.spot).trim())+"<br>🕒 "+esc(it.time)+"<br><br>"
    +(it.found?"주인을 찾아준 물건이에요 ✅":"청소 중 움직이지 않는 작은 물건을 발견해 사진으로 남겼어요. 확인하고 제자리에 두면 다음 청소가 더 편해요.");
  if(it.found){openModal(it.title,body);return;}
  openModal(it.title,body,{showCancel:true,cancelText:"닫기",confirmText:"✅ 찾았어요",onConfirm:()=>{
    it.found=true;closeModal();state.exp+=5;levelCheck();spawnEffect("🔍",7);
    addEvent("분실물 확인","'"+it.desc+"' 을(를) 확인했어요.","발견 기록");
    renderFound();render();showToast("분실물을 확인했어요! 로보킹이 기뻐해요.");
  }});
}
function openFoundMapBig(){
  const today=lostItems[0];if(!today)return;
  openModal("발견 위치","<div class='found-map' style='height:230px;margin-bottom:10px'>"+foundMapSvg(today)+"</div>📍 <b>"+esc((today.place+" "+today.spot).trim())+"</b> · "+esc(today.time)+"<br>"+esc(today.desc));
}

const missionDefs=[
  {key:"clean",icon:"🧹",name:"청소 마스터",unit:"회 청소",tiers:[{goal:10,coins:5},{goal:100,coins:20},{goal:1000,coins:100}],get:()=>totalCleanCount()},
  {key:"charge",icon:"🔋",name:"배터리 지킴이",unit:"회 맞춤 충전",tiers:[{goal:5,coins:5},{goal:30,coins:20},{goal:100,coins:60}],get:()=>state.acceptCount},
  {key:"lost",icon:"🔍",name:"탐정 로보킹",unit:"개 분실물 발견",tiers:[{goal:3,coins:5},{goal:20,coins:20},{goal:100,coins:80}],get:()=>lostItems.length},
  {key:"photo",icon:"📷",name:"반려동물 사진가",unit:"장 촬영",tiers:[{goal:3,coins:5},{goal:30,coins:20},{goal:100,coins:80}],get:()=>photos.length},
  {key:"learn",icon:"🏠",name:"우리 집 알아가기",unit:"회 학습",tiers:[{goal:1,coins:10}],get:()=>state.learnCount}
];
const medalFor=(def,i)=>def.tiers.length===1?"🏅":(["🥉","🥈","🥇"][i]||"🏅");
function claimableCount(){
  let n=0;
  missionDefs.forEach(d=>{const v=Number(d.get()||0);d.tiers.forEach((t,i)=>{if(v>=t.goal&&!state.claimedMissions[d.key+":"+i])n++;});});
  return n;
}
function renderMissions(){
  const list=$("missionList");if(!list)return;
  const claimable=claimableCount();
  const c=$("missionClaimable");if(c)c.textContent=claimable+"개";
  const dot=$("eventNavDot");if(dot)dot.style.display=claimable>0?"inline-block":"none";
  setHtml(list,missionDefs.map(d=>{
    const v=Number(d.get()||0);
    const rows=d.tiers.map((t,i)=>{
      const key=d.key+":"+i,reached=v>=t.goal,claimed=!!state.claimedMissions[key];
      const pct=Math.min(100,Math.round(v/t.goal*100));
      let btn;
      if(claimed)btn="<button type='button' class='tier-btn claimed' disabled>받음 ✓</button>";
      else if(reached)btn="<button type='button' class='tier-btn claim' data-action='claimMission' data-key='"+key+"'>+"+t.coins+" 코인 받기</button>";
      else btn="<button type='button' class='tier-btn' disabled>+"+t.coins+" 코인</button>";
      return "<div class='tier-row"+(reached?" reached":"")+"'><div class='tier-medal'>"+medalFor(d,i)+"</div>"
        +"<div class='tier-info'><div class='tier-goal'>"+t.goal.toLocaleString()+d.unit+"</div><div class='tier-track'><div class='tier-fill' style='width:"+pct+"%'></div></div></div>"+btn+"</div>";
    }).join("");
    return "<div class='mission-card'><div class='mission-head'><span class='m-icon'>"+d.icon+"</span><span class='m-name'>"+d.name+"</span><span class='m-count'>현재 "+v.toLocaleString()+d.unit+"</span></div>"+rows+"</div>";
  }).join(""));
}
function claimMission(el){
  const key=el.dataset.key||"";const parts=key.split(":");
  const d=missionDefs.find(x=>x.key===parts[0]);if(!d)return;
  const ti=Number(parts[1]);const t=d.tiers[ti];
  if(!t||state.claimedMissions[key])return;
  if(Number(d.get()||0)<t.goal){showToast("아직 목표에 도달하지 않았어요.");return}
  state.claimedMissions[key]=true;
  state.coins+=t.coins;state.exp+=10;levelCheck();
  const medal=medalFor(d,ti);
  spawnEffect(medal,10);render();
  addEvent("도전과제 달성",d.name+" · "+t.goal.toLocaleString()+d.unit+" 메달을 받았어요. +"+t.coins+" 코인","미션");
  showToast(medal+" "+d.name+" 메달 획득! +"+t.coins+" 코인");
}
function checkMissionUnlock(){
  const n=claimableCount();
  if(n>state.notifiedClaimable){spawnEffect("🏅",6);showToast("🏅 미션 달성! 이벤트 탭에서 메달과 코인을 받아요.");}
  state.notifiedClaimable=n;
}
function renderPhotos(){
  const grid=$("photoGrid"),empty=$("photoEmpty"),count=$("photoCount");
  if(!grid)return;
  if(count)count.textContent=photos.length;
  if(empty)empty.style.display=usingDemoPhotos?"block":"none";
  grid.innerHTML=photos.map((p,i)=>"<div class='photo-tile' data-action='photoOpen' data-idx='"+i+"'>"
    +(p.src?"<img src='"+p.src+"' alt='' loading='lazy'>":"<div class='ph-emoji'>"+(p.emoji||"🐾")+"</div>")
    +"<div class='photo-cap'>"+esc(p.title)+"<small>"+esc([p.place,p.time].filter(Boolean).join(" · "))+"</small></div></div>").join("");
}
function openPhoto(el){
  const p=photos[Number(el.dataset.idx)];if(!p)return;
  const visual=p.src?"<img class='modal-img' src='"+p.src+"' alt=''>":"<div class='modal-emoji'>"+(p.emoji||"🐾")+"</div>";
  openModal(p.title||"로보킹 사진",visual+(p.place?"📍 <b>"+esc(p.place)+"</b>":"")+(p.time?" · 🕒 "+esc(p.time):"")+(p.note?"<br>"+esc(p.note):"")+"<br><br>움직임을 감지했을 때 로보킹이 자동으로 찍어둔 사진이에요.");
}
function renderEvents(){
  const tabs={found:"evTabFound",mission:"evTabMission",photo:"evTabPhoto"};
  const panels={found:"evFoundPanel",mission:"evMissionPanel",photo:"evPhotoPanel"};
  Object.keys(tabs).forEach(k=>{const b=$(tabs[k]);if(b)b.classList.toggle("active",state.eventTab===k);const p=$(panels[k]);if(p)p.classList.toggle("hidden",state.eventTab!==k);});
  renderMissions();
}
function switchEventTab(tab){state.eventTab=tab;render();}

/* ============================================================
   원본 유지: 리워드 / 장식 / 모달 / 토스트 / 청소·충전 로직
   (새 페이지 연동을 위한 최소 훅만 추가: addEvent 태그, 카운터 증가)
   ============================================================ */
function renderAccessories(){
  const robot=$("robot");
  const head=$("robotHeadDeco");
  const aura=$("robotAuraDeco");
  const decal=$("robotBodyDeco");
  if(!robot || !head || !aura)return;

  robot.classList.toggle("has-custom-head",state.equippedItems.head && state.equippedItems.head!=="crown");

  const headItem=state.equippedItems.head;
  head.className="robot-accessory robot-head-deco";
  head.innerHTML="";
  if(headItem && headItem!=="crown"){
    if(headItem==="bunny"){
      head.classList.add("show","ears","bunny");
      head.innerHTML='<span class="robo-ear left"></span><span class="robo-ear right"></span>';
    }else if(headItem==="cat"){
      head.classList.add("show","ears","cat");
      head.innerHTML='<span class="robo-ear left"></span><span class="robo-ear right"></span>';
    }else{
      const headMap={ribbon:"🎀",hat:"🧢"};
      head.textContent=headMap[headItem]||"";
      head.classList.add("show",headItem);
    }
  }

  aura.className="robot-aura-deco";
  if(state.equippedItems.aura==="sparkle"){
    aura.innerHTML="<span>✨</span><span>✨</span><span>✨</span><span>✨</span>";
    aura.classList.add("show");
  }else{
    aura.innerHTML="";
  }

  // 이전 버전 호환용: 더 이상 몸통 스티커를 사용하지 않으므로 화면에서 숨깁니다.
  if(decal){
    decal.className="robot-accessory robot-body-deco";
    decal.textContent="";
  }
}

function renderReward(){
  $("levelText").textContent=state.level;
  $("expText").textContent=state.exp;
  $("expFill").style.width=state.exp+"%";

  const preview=$("levelRobotPreview");
  if(preview){
    const head=state.equippedItems.head;
    let html='<span class="preview-base">🤖</span>';
    if(head && head!=="crown"){
      if(head==="bunny" || head==="cat"){
        html+='<span class="preview-head ears '+head+'"><span class="p-ear left"></span><span class="p-ear right"></span></span>';
      }else{
        const headMap={ribbon:"🎀",hat:"🧢"};
        html+='<span class="preview-head '+head+'">'+(headMap[head]||'')+'</span>';
      }
    }
    if(state.equippedItems.aura==="sparkle")html+='<span class="preview-aura"><span class="a1">✨</span><span class="a2">✨</span><span class="a3">✨</span></span>';
    preview.innerHTML=html;
  }

  const itemTab=$("rewardTabItems");
  const couponTab=$("rewardTabCoupons");
  const itemPanel=$("rewardItemsPanel");
  const couponPanel=$("rewardCouponsPanel");
  if(itemTab)itemTab.classList.toggle("active",state.rewardTab==="items");
  if(couponTab)couponTab.classList.toggle("active",state.rewardTab==="coupons");
  if(itemPanel)itemPanel.classList.toggle("hidden",state.rewardTab!=="items");
  if(couponPanel)couponPanel.classList.toggle("hidden",state.rewardTab!=="coupons");

  updateRewardButton("ribbon","Ribbon");
  updateRewardButton("hat","Hat");
  updateRewardButton("bunny","Bunny");
  updateRewardButton("cat","Cat");
  updateRewardButton("sparkle","Sparkle");
  updateCouponButton("lg5","CouponLg5");
  updateCouponButton("cleanKit","CouponCleanKit");
  updateCouponButton("batteryCare","CouponBatteryCare");
  updateCouponButton("moveIn","CouponMoveIn");

  const foodBtn=$("btnFood");
  const foodStatus=$("statusFood");
  if(foodBtn){
    foodBtn.textContent=state.coins>=50?"50 코인":"50 코인 필요";
    foodBtn.classList.toggle("need-coins",state.coins<50);
  }
  if(foodStatus)foodStatus.textContent="보유 간식 "+state.food+"개";
}

function updateRewardButton(key,suffix){
  const item=shopItems[key];
  const btn=$("btn"+suffix);
  const card=$("card"+suffix);
  const status=$("status"+suffix);
  if(!item || !btn)return;
  const owned=Boolean(state.ownedItems[key]);
  const equipped=state.equippedItems[item.slot]===item.value;
  btn.classList.remove("owned","equipped","need-coins");
  if(card){card.classList.toggle("owned",owned);card.classList.toggle("equipped",equipped);}
  if(status)status.textContent=equipped?"장착 중":(owned?"보유 중":"");
  if(equipped){btn.textContent="해제하기";btn.classList.add("equipped");}
  else if(owned){btn.textContent="장착하기";btn.classList.add("owned");}
  else{
    btn.textContent=state.coins<item.cost ? item.cost+" 코인 필요" : item.cost+" 코인";
    if(state.coins<item.cost)btn.classList.add("need-coins");
  }
}

function updateCouponButton(key,suffix){
  const item=couponItems[key];
  const btn=$("btn"+suffix);
  const card=$("card"+suffix);
  const status=$("status"+suffix);
  if(!item || !btn)return;
  const count=Number(state.ownedCoupons[key]||0);
  btn.classList.remove("owned","equipped","need-coins");
  if(card)card.classList.toggle("owned",count>0);
  if(status)status.textContent=count>0?"보유 쿠폰 "+count+"장":"";
  if(state.coins<item.cost){
    btn.textContent=item.cost+" 코인 필요";
    btn.classList.add("need-coins");
  }
  else{btn.textContent=item.cost+" 코인으로 교환";}
}


function showToast(message){
  const toast=$("toast");toast.textContent=message;toast.classList.add("show");
  clearTimeout(window.toastTimer);
  window.toastTimer=setTimeout(()=>toast.classList.remove("show"),3500);
}
let modalConfirmHandler=closeModal;
function openModal(title,body,options={}){
  $("modalTitle").textContent=title;
  $("modalBody").innerHTML=body;
  const actions=$("modalActions");
  const cancelBtn=$("modalCancel");
  const confirmBtn=$("modalConfirm");
  const showCancel=Boolean(options.showCancel);
  actions.classList.toggle("single",!showCancel);
  cancelBtn.textContent=options.cancelText||"취소";
  confirmBtn.textContent=options.confirmText||"확인";
  modalConfirmHandler=typeof options.onConfirm==="function"?options.onConfirm:closeModal;
  $("modal").classList.add("show");
}
function closeModal(){$("modal").classList.remove("show")}

function spawnEffect(symbol,count=7){
  const layer=$("effectLayer");
  for(let i=0;i<count;i++){
    const p=document.createElement("span");
    p.className="effect";p.textContent=symbol;
    p.style.setProperty("--move-x",Math.round(Math.random()*160-80)+"px");
    p.style.setProperty("--rotate",Math.round(Math.random()*100-50)+"deg");
    p.style.left=(42+Math.random()*16)+"%";
    p.style.animationDelay=(Math.random()*.22)+"s";
    layer.appendChild(p);setTimeout(()=>p.remove(),1500);
  }
}
function pulseRobot(){const robot=$("robot");robot.classList.remove("tap");void robot.offsetWidth;robot.classList.add("tap");setTimeout(()=>robot.classList.remove("tap"),650)}
function levelCheck(){if(state.exp>=100){state.exp-=100;state.level+=1;spawnEffect("⭐",10);showToast("레벨 업! Lv."+state.level)}}
// 케어 기록(2번째 탭)에 실시간으로 쌓입니다. tag는 "어떻게 수명을 지켰는지" 한 줄 라벨입니다.
function addEvent(title,description,tag){
  const list=$("eventList");
  if(!list)return;
  const now=new Date();
  const hh=String(now.getHours()).padStart(2,"0"),mm=String(now.getMinutes()).padStart(2,"0");
  const row=document.createElement("div");row.className="event-item";
  row.innerHTML='<div class="event-time">'+hh+':'+mm+'</div><div class="event-content"><strong>'+esc(title)+(tag?'<span class="event-tag">'+esc(tag)+'</span>':'')+'</strong><span>'+description+'</span></div>';
  list.prepend(row);
  while(list.children.length>30)list.removeChild(list.lastChild);
}

function petRobot(){if(state.cleaning){showToast("청소가 끝난 후 로보킹을 쓰다듬어 주세요.");return}state.heart=Math.min(100,state.heart+2);state.exp+=1;pulseRobot();spawnEffect("💖",7);levelCheck();render();showToast("로보킹의 기분이 좋아졌어요.")}
function feedRobot(){if(state.food<=0){showToast("음식이 부족해요. 리워드에서 구매해 주세요.");return}state.food-=1;state.soc+=12;state.exp+=8;pulseRobot();spawnEffect("⚡",8);levelCheck();render();showToast("배터리가 12% 회복되었습니다.")}
function playRobot(){if(state.soc<5){showToast("배터리가 부족해서 놀 수 없어요.");return}state.soc-=3;state.exp+=5;pulseRobot();spawnEffect("💖",8);levelCheck();render();showToast("로보킹의 친밀도와 경험치가 올랐어요.")}
function trainRobot(){if(state.soc<8){showToast("훈련 전에 충전이 필요해요.");return}state.soc-=6;state.health=Math.min(100,state.health+3);state.exp+=12;pulseRobot();spawnEffect("✨",8);levelCheck();render();showToast("로보킹이 훈련을 완료했습니다.")}
function takePhoto(){pulseRobot();spawnEffect("📸",5);state.eventTab="photo";switchPage("eventPage");showToast("로보킹 사진첩을 열었어요.")}
function decorateRobot(){
  switchPage("rewardPage");
  showToast("리워드에서 아이템을 사면 로보킹에게 계속 장착돼요.");
}

function showStatus(){
  if(!state.profileReady){
    openModal("먼저 우리 집을 배울게요","아직 로보킹이 우리 집을 잘 몰라요.<br><br>1회차 학습 청소를 시작하면 방 구조와 바닥 상태를 기억하고, 다음부터 더 똑똑하게 청소를 준비할 수 있어요.");
    return;
  }
  if(!state.predicted){
    openModal("우리 집을 기억했어요","1회차 학습 청소가 끝났어요.<br><br>이제 오늘 청소 조건을 고르고 <b>오늘 청소 준비하기</b>를 눌러 주세요.<br>로보킹이 알아서 필요한 만큼 준비할게요.");
    return;
  }
  const scopeText=state.selectedScope==="home"?"집 전체":state.selectedLabel;
  const zoneInfo=state.selectedScope==="zone"?"<br>바닥: <b>"+(state.floorType||"정보 없음")+"</b><br>상태: <b>"+(state.dirtLevel||"평소")+"</b>":"";
  const readyText=state.soc>=state.targetSoc?"지금 바로 출동할 수 있어요.":"잠깐만 충전하면 출동할 수 있어요.";
  openModal("오늘 청소 준비 완료",scopeText+" 청소를 준비했어요."+zoneInfo+"<br><br>오늘 조건: <b>"+state.cleanModeLabel+" · "+state.intensityLabel+" · "+state.todayStateLabel+"</b><br><br>"+readyText+"<br>로보킹이 배터리를 아끼면서 청소할게요.");
}

function getRemainingCleaningSoc(){
  const total=Math.max(0,Number(state.requiredSoc||0));
  let remaining=Number(state.cleaningRemainingSoc);
  if(!Number.isFinite(remaining) || remaining<=0 || state.progress>=100 || state.missionDone){
    remaining=total;
  }
  return Math.round(Math.min(Math.max(remaining,0),total)*10)/10;
}

function resetCleaningMissionPlan(){
  state.cleaningRemainingSoc=Math.max(0,Number(state.requiredSoc||0));
  state.cleaningSegmentIndex=0;
  state.splitCleaning=state.cleaningRemainingSoc>MAX_SINGLE_PASS_USE;
  state.progress=0;
  state.missionDone=false;
}

function showSplitCleaningModal(){
  state.targetSoc=MAX_CHARGE_SOC;
  state.splitCleaning=true;
  render();
  const body="청소할 양이 많아서<br>"
    +"한 번에 무리하면 로보킹이 금방 지칠 수 있어요.<br><br>"
    +"배터리를 아끼기 위해<br>"
    +"잠깐 쉬어가며 이어서 청소할게요.";
  openModal("이번 청소는 나눠서 할게요",body,{
    showCancel:true,
    cancelText:"취소",
    confirmText:state.soc<MAX_CHARGE_SOC?"충전하고 시작":"청소 시작",
    onConfirm:()=>{
      closeModal();
      if(state.soc<MAX_CHARGE_SOC){
        chargeRobot(true);
      }else{
        startCleaning();
      }
    }
  });
}

function showReserveChargeModal(autoStartAfterCharge=false){
  const remaining=getRemainingCleaningSoc();
  const needed=targetFromRequired(remaining);
  state.targetSoc=needed;
  render();
  const body=state.selectedLabel+" 청소를 바로 시작하기엔<br>로보킹의 힘이 조금 부족해요.<br><br>잠깐 충전하고 나면<br>청소를 더 편하게 마칠 수 있어요.<br><br>필요한 만큼만 채우고 바로 출발할게요!";
  openModal("먼저 힘을 채울게요",body,{
    showCancel:true,
    cancelText:"취소",
    confirmText:"충전하고 시작",
    onConfirm:()=>{
      closeModal();
      switchPage("homePage");
      chargeRobot(autoStartAfterCharge);
    }
  });
}

function showChargeChoiceModal(autoStartAfterCharge=false){
  const remaining=getRemainingCleaningSoc();
  const needed=targetFromRequired(remaining);
  state.targetSoc=needed;
  render();
  const scopeText=state.selectedScope==="zone"?state.selectedLabel+"은 <b>"+(state.floorType||"바닥 정보")+"</b> 바닥이라 조금 더 힘이 필요해요.<br><br>":"";
  const body=scopeText+"이번 청소를 끝까지 편하게 마치려면<br>로보킹이 힘을 조금 더 채우면 좋아요.<br><br>필요한 만큼만 충전하고<br>바로 청소를 시작할게요.";
  openModal("아직 배가 조금 고파요!",body,{
    showCancel:true,
    cancelText:"취소",
    confirmText:"충전하고 시작",
    onConfirm:()=>{
      closeModal();
      switchPage("homePage");
      chargeRobot(autoStartAfterCharge);
    }
  });
}


function makeAggregateScenario(zones,label,mode){
  const areaSum=zones.reduce((sum,z)=>sum+Number(z.cleaningAreaM2||0),0);
  const requiredSum=zones.reduce((sum,z)=>sum+Number(z.requiredSoc||0),0);
  const highest=zones.slice().sort((a,b)=>getZoneConditionScore(b.zone)-getZoneConditionScore(a.zone))[0] || activeRun.home;
  return {
    scope:"home",
    zone:null,
    label:label,
    globalRunId:activeRun.home.globalRunId,
    areaPyung:activeRun.home.areaPyung,
    cleaningAreaM2:Math.round(areaSum*10)/10,
    requiredSoc:Math.round(requiredSum*10)/10,
    targetSoc:targetFromRequired(requiredSum),
    modelName:activeRun.home.modelName,
    cleaningType:activeRun.home.cleaningType,
    cleaningTypeCode:activeRun.home.cleaningTypeCode,
    mopEnabled:activeRun.home.mopEnabled,
    obstacleLevel:activeRun.home.obstacleLevel,
    obstacleLevelCode:activeRun.home.obstacleLevelCode,
    floorType:highest.floorType || activeRun.home.floorType,
    dirtLevel:highest.dirtLevel || activeRun.home.dirtLevel,
    dirtCode:highest.dirtCode || activeRun.home.dirtCode,
    suctionMode:highest.suctionMode || activeRun.home.suctionMode,
    suctionCode:highest.suctionCode || activeRun.home.suctionCode,
    cleanModeChoice:state.cleanModeChoice,
    cleanModeLabel:state.cleanModeLabel,
    intensityChoice:state.intensityChoice,
    intensityLabel:state.intensityLabel,
    todayStateChoice:state.todayStateChoice,
    todayStateLabel:state.todayStateLabel,
    matchNote:mode==="dirty"?"먼지가 많은 곳만 골라 준비":"금지구역은 빼고 알아서 준비",
    matchBasis:"우리 집 매핑 정보 반영"
  };
}
function prepareScenarioAndShow(scenario,message,tone="done"){
  state.chargePurpose='current';
  state.nextHomeReady=false;
  syncScenarioToState(scenario);
  state.predicted=true;
  state.predicting=false;
  state.chargeComplete=false;
  state.cleaningRemainingSoc=Number(state.requiredSoc||0);
  state.progress=0;
  render();
  const canNow=state.soc>=state.targetSoc;
  $("speech").innerHTML="<strong style='color:#2f8b3a'>준비 완료!</strong><br>"+(canNow?"바로 출동할 수 있어요.":"잠깐 충전하고 출발할게요.");
  setModeChipText("✅ "+state.selectedLabel+" 준비 완료");
  setGuide(message,canNow?tone:"warning");
  showToast(message.replace(/<[^>]*>/g,""));
}
function aiAutoClean(){
  if(!state.profileReady){showToast("먼저 1회차 학습 청소를 시작해 주세요.");return}
  if(state.mapping||state.cleaning||state.charging){showToast("진행 중인 작업이 끝난 뒤 선택할 수 있어요.");return}

  const cleanable=getCleanableZones();
  const allZones=cleanable.map(z=>Number(z.zone));
  if(!cleanable.length){showToast("청소할 수 있는 영역이 없어요. 금지구역을 줄여주세요.");return}

  state.mapMode="view";
  state.smartCleanMode="auto";
  state.selectedScope="home";
  state.selectedZone=null;
  state.manualReady=false;
  state.manualKey="";
  state.selectedDirtyZones=[];
  state.completedZones=[];
  state.currentCleaningZone=null;
  state.cleaningZones=allZones;

  // 핵심: AI 자동청소는 항상 전체 zone SOC 합산값을 사용합니다.
  // 금지구역이 있으면 그 구역만 제외하고 합산합니다.
  const scenario=makeAggregateScenario(cleanable,"AI 자동청소","auto");
  scenario.scope="home";
  scenario.label="AI 자동청소";
  scenario.requiredSoc=Math.round(cleanable.reduce((sum,z)=>sum+Number(z.requiredSoc||0),0)*10)/10;
  scenario.targetSoc=targetFromRequired(scenario.requiredSoc);
  scenario.matchNote=(state.noGoZones&&state.noGoZones.length)
    ? "금지구역을 제외한 전체 청소"
    : "전체 구역 자동청소";
  scenario.matchBasis=allZones.length+"개 영역 모두 반영";

  prepareScenarioAndShow(
    scenario,
    "AI 자동청소 준비 완료! "+allZones.length+"개 영역을 모두 청소할게요.",
    "done"
  );
}
function dirtyOnlyClean(){
  if(!state.profileReady){showToast("먼저 1회차 학습 청소를 시작해 주세요.");return}
  if(state.mapping||state.cleaning||state.charging){showToast("진행 중인 작업이 끝난 뒤 선택할 수 있어요.");return}
  state.mapMode="view";
  state.smartCleanMode="dirty";
  state.manualReady=false;
  state.manualKey="";

  const cleanable=getCleanableZones();
  if(!cleanable.length){showToast("청소할 수 있는 영역이 없어요. 금지구역을 줄여주세요.");return}

  const sorted=cleanable.slice().sort((a,b)=>getZoneConditionScore(b.zone)-getZoneConditionScore(a.zone));
  const count=Math.min(Math.max(1,Math.ceil(sorted.length*0.35)),3);
  const picked=sorted.slice(0,count).sort((a,b)=>Number(a.zone)-Number(b.zone));
  state.selectedDirtyZones=picked.map(z=>Number(z.zone));
  state.cleaningZones=state.selectedDirtyZones.slice();
  state.completedZones=[];
  state.currentCleaningZone=null;

  const scenario=makeAggregateScenario(picked,"더러운 곳만","dirty");
  prepareScenarioAndShow(scenario,"더 신경 쓸 곳만 골랐어요. 이 영역부터 깨끗하게 청소할게요.","done");
}
function toggleNoGoMode(){
  if(!state.profileReady){showToast("먼저 1회차 학습 청소를 시작해 주세요.");return}
  if(state.mapping||state.cleaning||state.charging){showToast("진행 중인 작업이 끝난 뒤 설정할 수 있어요.");return}

  state.mapMode = state.mapMode==="noGo" ? "view" : "noGo";
  if(state.mapMode==="noGo"){
    state.smartCleanMode="auto";
    state.selectedDirtyZones=[];
    state.completedZones=[];
    state.currentCleaningZone=null;
    state.cleaningZones=getCleanableZones().map(z=>Number(z.zone));
    setGuide("청소하지 않을 영역을 지도에서 눌러주세요. 다시 누르면 해제돼요.","warning");
    showToast("금지구역 설정: 지도에서 제외할 영역을 눌러주세요.");
  }else{
    const count=(state.noGoZones||[]).length;
    setGuide(count>0?"금지구역 "+count+"곳을 빼고 청소할 수 있어요.":"금지구역 설정을 마쳤어요.","done");
    showToast(count>0?"금지구역 "+count+"곳을 저장했어요.":"금지구역 설정을 마쳤어요.");
  }
  render();
}
function handleMapZoneTap(element){
  const zoneNo=Number(element && element.dataset ? element.dataset.zone : element);
  if(!zoneNo)return;
  if(!state.profileReady){showToast("1회차 학습 후 지도에서 선택할 수 있어요.");return}

  if(state.mapMode==="noGo"){
    const list=state.noGoZones || [];
    const idx=list.indexOf(zoneNo);
    if(idx>=0){
      list.splice(idx,1);
      showToast(zoneNo+"번 영역 금지구역을 해제했어요.");
    }else{
      list.push(zoneNo);
      list.sort((a,b)=>a-b);
      showToast(zoneNo+"번 영역은 청소하지 않을게요.");
    }
    state.noGoZones=list;
    state.predicted=false;
    state.manualReady=false;
    state.manualKey="";
    state.smartCleanMode="auto";
    state.selectedDirtyZones=[];
    state.completedZones=[];
    state.currentCleaningZone=null;
    state.cleaningZones=getCleanableZones().map(z=>Number(z.zone));
    render();
    return;
  }

  // 보기 모드에서는 구역을 누르면 해당 영역만 빠르게 준비합니다.
  const zone=getScenario("zone",zoneNo);
  if(!zone){showToast("이 영역 정보를 찾지 못했어요.");return}
  const choices=getPredictionChoices("zone",zoneNo);
  const matchedScenario=findMlScenarioFromChoices(choices);
  matchedScenario.label=zoneNo+"번 영역";
  state.smartCleanMode="zone";
  state.selectedDirtyZones=[];
  state.cleaningZones=[Number(zoneNo)];
  state.completedZones=[];
  state.currentCleaningZone=null;
  prepareScenarioAndShow(matchedScenario,zoneNo+"번 영역만 청소할 준비를 마쳤어요.","done");
}




function manualCleanAndGo(){
  if(state.cleaning){showToast("이미 청소 중이에요.");return}
  if(state.charging){showToast("충전이 끝나면 바로 출발할게요.");return}
  if(state.mapping){showToast("집을 다 배운 뒤 청소할 수 있어요.");return}
  if(!state.profileReady){
    setGuide("먼저 1회차 학습 청소로 우리 집을 알려주세요.","warning");
    showToast("먼저 로보킹에게 우리 집을 알려주세요.");
    return;
  }

  const key=getManualSelectionKey();
  const manualReady=state.predicted && state.smartCleanMode==="manual" && state.manualReady && state.manualKey===key;

  // 조건이 아직 적용되지 않았거나 바뀌었다면:
  // 1) 선택 조건으로 청소 준비
  // 2) 준비가 끝나면 자동으로 충전/청소까지 이어짐
  if(!manualReady){
    predictSocFromConditions(true);
    return;
  }

  // 이미 같은 조건으로 준비되어 있으면 바로 충전/청소 실행
  executeTopClean();
}

function executeTopClean(){
  if(state.cleaning){showToast("이미 청소 중이에요.");return}
  if(state.charging){showToast("충전이 끝난 뒤 바로 출발할게요.");return}
  if(state.mapping){showToast("집을 다 배운 뒤 청소할 수 있어요.");return}
  if(!state.profileReady){
    setGuide("먼저 1회차 학습 청소로 우리 집을 알려주세요.","warning");
    showToast("먼저 로보킹에게 우리 집을 알려주세요.");
    return;
  }

  // 사용자가 따로 선택하지 않으면 가장 쉬운 기본값인 AI 자동청소로 준비합니다.
  if(!state.predicted){
    aiAutoClean();
    if(!state.predicted)return;
  }

  if(state.soc < state.targetSoc){
    if(Number(state.requiredSoc||0)>MAX_SINGLE_PASS_USE && state.soc<MAX_CHARGE_SOC){
      showSplitCleaningModal();
    }else{
      chargeRobot(true);
    }
  }else{
    state.robotMotion='idle';
    startCleaning();
  }
}


function getCleaningZonesForCurrentPlan(){
  if(state.smartCleanMode==="auto"){
    return getAllCleanableZoneNumbers();
  }
  const nums=getPlannedZoneNumbers();
  if(nums.length)return nums;
  if(state.selectedScope==="zone" && state.selectedZone)return [Number(state.selectedZone)];
  return getAllCleanableZoneNumbers();
}
function updateCleaningZoneProgress(percent){
  const zones=state.cleaningZones && state.cleaningZones.length ? state.cleaningZones : getCleaningZonesForCurrentPlan();
  if(!zones.length){
    state.currentCleaningZone=null;
    state.completedZones=[];
    return;
  }
  const ratio=clamp(Number(percent||0),0,99)/100;
  const idx=Math.min(zones.length-1,Math.floor(ratio*zones.length));
  state.currentCleaningZone=zones[idx];
  state.completedZones=zones.slice(0,idx);
}
function finishCleaningZoneProgress(){
  const zones=state.cleaningZones && state.cleaningZones.length ? state.cleaningZones : getCleaningZonesForCurrentPlan();
  state.completedZones=zones.slice();
  state.currentCleaningZone=null;
}
function clearCleaningZoneProgress(){
  state.cleaningZones=[];
  state.currentCleaningZone=null;
  state.completedZones=[];
}


function startCleaning(){
  if(state.cleaning){showToast("이미 청소 중이에요.");return}
  if(state.charging){showToast("충전이 끝난 후 청소할게요.");return}
  if(state.mapping){showToast("1회차 학습이 끝난 뒤 청소할 수 있어요.");return}
  if(!state.profileReady){
    setGuide("아직 로보킹이 우리 집을 잘 몰라요. 먼저 1회차 학습 청소를 시작해 주세요.","warning");
    showToast("먼저 로보킹에게 우리 집을 알려주세요.");
    $("speech").innerHTML="<strong style='color:#ef8c32'>학습이 먼저예요</strong><br>집 정보를 저장한 뒤 청소할 수 있어요.";
    switchPage("homePage");
    return;
  }
  if(!state.predicted){
    showToast("청소 전 오늘 청소 준비하기를 먼저 눌러주세요.");
    $("speech").innerHTML="<strong style='color:#2f8b3a'>청소 준비가 필요해요</strong><br>오늘 상태를 먼저 알려주세요.";
    switchPage("homePage");
    return;
  }

  const totalRequired=Math.max(0,Number(state.requiredSoc||0));
  if(totalRequired<=0){showToast("오늘 청소 준비를 다시 실행해 주세요.");return}

  if(state.missionDone || state.progress>=100){
    resetCleaningMissionPlan();
  }

  let remaining=getRemainingCleaningSoc();

  // 90% 상한과 15% 잔량 기준으로 한 번에 끝낼 수 없는 경우에만 분할 청소 안내를 띄웁니다.
  if(remaining>MAX_SINGLE_PASS_USE && state.soc<MAX_CHARGE_SOC){
    showSplitCleaningModal();
    return;
  }

  const neededStart=targetFromRequired(remaining);
  state.targetSoc=neededStart;

  // 청소 시작 전 Reserve 배터리 Guard
  if(remaining<=MAX_SINGLE_PASS_USE && state.soc<neededStart){
    showReserveChargeModal(true);
    return;
  }

  if(state.soc<=MIN_RESERVE_SOC){
    showReserveChargeModal(true);
    return;
  }

  const availableUse=Math.max(0,Number(state.soc||0)-MIN_RESERVE_SOC);
  let segmentUse=remaining;
  let segmentWillComplete=true;

  // 분할 청소 중 첫 구간: 현재 배터리에서 15%를 남길 수 있는 만큼만 청소
  if(remaining>availableUse){
    segmentUse=availableUse;
    segmentWillComplete=false;
  }

  if(segmentUse<=0){
    showReserveChargeModal(true);
    return;
  }

  state.cleaning=true;
  // 바로 청소 가능한 경우에는 스테이션 복귀/출발 모션 없이 즉시 청소를 시작합니다.
  // 스테이션 출발 모션은 실제 충전 후 자동 출발할 때만 chargeRobot()에서 실행합니다.
  state.robotMotion='idle';
  state.cleaningZones=getCleaningZonesForCurrentPlan();
  updateCleaningZoneProgress(state.progress||0);
  state.chargeComplete=false;
  state.missionDone=false;
  const startSoc=Number(state.soc||0);
  const startProgress=Number(state.progress||0);
  const progressGain=Math.max(1,Math.round(segmentUse/totalRequired*100));
  const endProgress=segmentWillComplete?100:Math.min(99,startProgress+progressGain);
  const endSoc=Math.max(MIN_RESERVE_SOC,Math.round((startSoc-segmentUse)*10)/10);
  state.cleaningSegmentIndex+=1;
  // 지도 위 로봇 아이콘을 부드럽게 움직이기 위한 시간 기반 진행 정보
  state.cleanAnim={startedAt:Date.now(),duration:20*320,fromProgress:startProgress,toProgress:endProgress};
  render();
  startMapRobotAnim();
  setGuide(state.selectedLabel+" 청소를 시작했어요. 로보킹이 배터리를 아끼면서 깨끗하게 청소할게요.","normal");
  showToast("청소 시작! 로보킹이 배터리를 아끼며 청소해요.");

  let step=0;
  const totalSteps=20;
  const timer=setInterval(()=>{
    step+=1;
    const ratio=step/totalSteps;
    state.progress=Math.round(startProgress+(endProgress-startProgress)*ratio);
    updateCleaningZoneProgress(state.progress);
    state.soc=Math.max(MIN_RESERVE_SOC,Math.round((startSoc-segmentUse*ratio)*10)/10);
    state.temperature=Math.min(36,state.temperature+.25);
    render();

    if(state.soc<=CRITICAL_DOCK_SOC && !segmentWillComplete){
      step=totalSteps;
    }
    if(state.cleanAnim && step>=totalSteps)state.cleanAnim.duration=Math.max(1,Date.now()-state.cleanAnim.startedAt);

    if(step>=totalSteps){
      clearInterval(timer);
      state.cleaning=false;
      state.cleanAnim=null;
      state.temperature=29;
      state.soc=endSoc;
      const newRemaining=Math.max(0,Math.round((remaining-segmentUse)*10)/10);
      state.cleaningRemainingSoc=newRemaining;

      if(newRemaining>0.2){
        state.progress=endProgress;
        updateCleaningZoneProgress(state.progress);
        state.robotMotion='returning';
        state.targetSoc=targetFromRequired(newRemaining);
        // [부품케어 탭 연동] 15% 잔량 보호 횟수 + 케어 기록
        state.reserveGuardCount+=1;
        addEvent("잠깐 쉬어가기",state.selectedLabel+" 청소 중 배터리 15%가 되어 스스로 도킹했어요. 잠깐 충전 후 남은 곳을 이어서 청소해요.","잔량 15% 보호");
        render();
        $("speech").innerHTML="<strong style='color:#ef8c32'>잠깐 쉬어갈게요!</strong><br>조금만 쉬고 다시 힘낼게요.";
        setGuide("로보킹이 조금 지쳤어요. 잠깐 충전하고 남은 곳을 이어서 청소할게요.","warning");
        showToast("잠깐 충전하고 남은 곳을 이어서 청소할게요.");
        setTimeout(()=>openModal("잠깐 쉬어갈게요!","제가 조금 지쳤어요.<br>잠깐 충전하고 나면<br>남은 곳도 다시 힘내서 청소할게요!<br><br>지금 배터리: <b>"+fmtSoc(state.soc)+"%</b>",{
          showCancel:true,
          cancelText:"나중에",
          confirmText:"충전하고 이어서",
          onConfirm:()=>{closeModal();chargeRobot(true);}
        }),450);
        return;
      }

      state.cleaningRemainingSoc=0;
      state.progress=100;
      finishCleaningZoneProgress();
      state.robotMotion='idle';
      state.missionDone=true;
      state.celebrating=true;
      state.cleanCount+=1;
      state.coins+=50;
      state.exp+=20;
      state.area=Math.round((state.area||0)+(state.cleaningAreaM2||0));
      state.average=Math.round((state.average+Math.max(15,Math.round(state.requiredSoc*1.4)))/2);
      levelCheck();
      // [부품케어 탭 연동] 배터리 절약 기록
      addEvent(state.selectedLabel+" 청소 완료","배터리 "+fmtSoc(totalRequired)+"%만 사용해 청소를 마쳤어요. 15% 이상 남겨 배터리에 무리를 주지 않았어요.","배터리 절약");
      spawnEffect("🎉",15);spawnEffect("⭐",9);
      render();
      $("speech").innerHTML="<strong style='color:#2f8b3a'>청소 완료!</strong><br>+50코인을 받았어요.";
      setModeChipText("🏆 "+state.selectedLabel+" 완료 · +50코인");
      setGuide("청소 완료! 배터리를 아껴 쓰며 마무리했어요. 보상으로 +50코인과 경험치를 받았어요.","done");
      showToast("청소 완료! 로보킹이 +50코인을 가져왔어요.");
      setTimeout(()=>{
        state.celebrating=false;
        clearCleaningZoneProgress();
        render();
        // 청소가 끝나면 다음 집 전체 청소에 필요한 만큼 미리 충전합니다.
        // 소형/중형/대형별 4/6/8개 zone의 requiredSoc 합산값을 기준으로 목표 충전량을 정합니다.
        setTimeout(prepareNextWholeHomeCharge,350);
      },3600);
      // [이벤트 탭 연동] 미션 달성 알림
      setTimeout(checkMissionUnlock,4200);
    }
  },320);
}

function prepareNextWholeHomeCharge(){
  if(state.mapping || state.cleaning || state.charging)return;
  if(!state.profileReady || !activeRun)return;

  const wholeRequired=getWholeHomeRequiredSoc();
  if(wholeRequired<=0)return;

  const wholeTarget=getWholeHomeTargetSoc();
  const wholeZones=getWholeHomeZones();

  state.nextHomeRequiredSoc=wholeRequired;
  state.nextHomeTargetSoc=wholeTarget;
  state.targetSoc=wholeTarget;
  state.requiredSoc=wholeRequired;
  state.cleaningRemainingSoc=wholeRequired;
  state.progress=0;
  state.selectedScope="home";
  state.selectedZone=null;
  state.selectedLabel="다음 전체 청소";
  state.smartCleanMode="auto";
  state.cleaningZones=wholeZones.map(z=>Number(z.zone));
  state.selectedDirtyZones=[];
  state.completedZones=[];
  state.currentCleaningZone=null;
  state.predicted=true;
  state.chargePurpose="nextHome";

  if(state.soc>=wholeTarget){
    state.nextHomeReady=true;
    state.chargeComplete=true;
    render();
    const msg="다음 전체 청소도 바로 할 수 있게 준비해뒀어요.";
    const speech=$("speech");
    if(speech)speech.innerHTML="<strong style='color:#2f8b3a'>다음 청소 준비 완료!</strong><br>필요한 만큼 채워뒀어요.";
    const chip=$("modeChip");
    if(chip)chip.textContent="✅ 다음 전체 청소 준비 완료";
    setGuide(msg,"done");
    showToast(msg);
    setTimeout(()=>{state.chargeComplete=false;render();},2600);
    return;
  }

  state.nextHomeReady=false;
  setGuide("청소가 끝났어요. 다음 전체 청소를 위해 필요한 만큼 미리 충전해둘게요.","charging");
  showToast("다음 청소를 위해 미리 힘을 채울게요.");
  chargeRobot(false,"nextHome");
}

function chargeRobot(autoStart=false,purpose='current'){
  if(state.cleaning){showToast("청소가 끝난 후 충전할 수 있어요.");return}
  if(state.charging){showToast("이미 충전 중이에요.");return}
  state.chargePurpose=purpose || 'current';
  const isNextHomeCharge=state.chargePurpose==='nextHome';
  if(state.soc>=state.targetSoc){
    if(autoStart){state.robotMotion='idle';setTimeout(startCleaning,250);return}
    state.chargeComplete=true;
    if(isNextHomeCharge)state.nextHomeReady=true;
    render();
    const speech=$("speech");
    const chip=$("modeChip");
    if(isNextHomeCharge){
      if(speech)speech.innerHTML="<strong style='color:#2f8b3a'>다음 청소 준비 완료!</strong><br>필요한 만큼 채워뒀어요.";
      if(chip)chip.textContent="✅ 다음 전체 청소 준비 완료";
      setGuide("다음 전체 청소도 바로 할 수 있게 준비해뒀어요.","done");
      showToast("다음 청소 준비 완료! 필요한 만큼 채워뒀어요.");
    }else{
      if(speech)speech.innerHTML="<strong>배불러요!</strong><br>이제 "+state.selectedLabel+" 청소가 가능해요.";
      if(chip)chip.textContent="💖 출동 준비 완료";
      setGuide("이미 충분히 준비됐어요. 바로 청소를 시작할 수 있어요.","done");
      showToast("이미 충분히 준비됐어요. 바로 출동할 수 있어요.");
    }
    setTimeout(()=>{state.chargeComplete=false;render()},2600);
    return;
  }
  closeModal();
  switchPage("homePage");
  state.charging=true;
  state.robotMotion='returning';
  state.chargeComplete=false;
  render();
  if(isNextHomeCharge){
    setGuide("다음 전체 청소를 위해 로보킹이 스테이션에서 미리 힘을 채우고 있어요.","charging");
    showToast("다음 청소를 위해 미리 충전할게요.");
  }else{
    setGuide("로보킹이 스테이션으로 돌아가고 있어요. 필요한 만큼만 충전하고 출발할게요.","charging");
    showToast("스테이션으로 돌아가 힘을 채울게요.");
  }
  setTimeout(()=>{state.robotMotion='docked';render();},950);
  setTimeout(()=>{
  const timer=setInterval(()=>{
    state.soc=Math.min(state.targetSoc,state.soc+2);
    state.temperature=Math.min(32,state.temperature+.1);
    spawnEffect("⚡",2);
    render();
    if(state.soc>=state.targetSoc){
      clearInterval(timer);
      state.charging=false;
      state.robotMotion='docked';
      state.temperature=29;
      state.acceptCount+=1;
      if(isNextHomeCharge)state.nextHomeReady=true;
      // [부품케어 탭 연동] 덜 채운 충전량 누적 + 수명 보호 기록
      state.savedChargePct+=Math.max(0,100-state.targetSoc);
      state.chargeComplete=true;
      if(isNextHomeCharge){
        addEvent("다음 청소 준비 완료","집 전체 청소에 필요한 "+state.targetSoc+"%까지만 미리 채워뒀어요. 완충하지 않고 필요한 만큼만 준비했어요.","다음 청소 준비");
      }else{
        addEvent("맞춤 충전 완료",state.selectedLabel+" 청소에 필요한 "+state.targetSoc+"%까지만 채우고 멈췄어요. 완충 대비 "+(100-state.targetSoc)+"% 덜 채워 과충전을 막았어요.","수명 보호");
      }
      spawnEffect("💖",12);
      spawnEffect("✨",8);
      render();
      const speech=$("speech");
      const chip=$("modeChip");
      if(isNextHomeCharge){
        if(speech)speech.innerHTML="<strong style='color:#2f8b3a'>다음 청소 준비 완료!</strong><br>필요한 만큼 채워뒀어요.";
        if(chip)chip.textContent="✅ 다음 전체 청소 준비 완료";
        setGuide("다음 전체 청소도 바로 시작할 수 있게 미리 준비해뒀어요.","done");
        showToast("다음 청소 준비 완료! 필요한 만큼 채워뒀어요.");
      }else{
        if(speech)speech.innerHTML="<strong>배불러요!</strong><br>출동할 준비가 됐어요!";
        if(chip)chip.textContent="💖 충전 완료 · 출동 준비";
        setGuide("충전 완료! 로보킹이 곧 바로 출동할게요.","done");
        showToast("충전 완료! 이제 로보킹이 출동할 수 있어요.");
      }
      setTimeout(()=>{state.chargeComplete=false;render()},3200);
      if(autoStart){setTimeout(()=>{state.robotMotion='departing';render();setTimeout(()=>{state.robotMotion='idle';startCleaning();},850)},900)}
      else setTimeout(checkMissionUnlock,3600);
    }
  },150);
  },1050);
}
function buyFood(){
  if(state.coins<50){showToast("코인이 조금 부족해요. 청소 미션으로 코인을 모아보세요.");return}
  state.coins-=50;
  state.food+=1;
  render();
  showToast("냠냠! 에너지 간식 1개를 챙겼어요. 필요할 때 먹여주세요.");
}
function handleRewardItem(key){
  const item=shopItems[key];
  if(!item)return;
  if(!state.ownedItems[key]){
    if(state.coins<item.cost){
      showToast(item.name+"을(를) 데려오려면 코인이 조금 더 필요해요.");
      return;
    }
    state.coins-=item.cost;
    state.ownedItems[key]=true;
    state.equippedItems[item.slot]=item.value;
    saveCloset();
    switchPage("homePage");
    setTimeout(()=>{spawnEffect(item.icon,10);showToast(item.message);render();},250);
    render();
    return;
  }
  const isEquipped=state.equippedItems[item.slot]===item.value;
  if(isEquipped){
    state.equippedItems[item.slot]=(item.slot==="head"?"crown":null);
    saveCloset();
    render();
    showToast(item.name+"을(를) 잠시 벗겨뒀어요.");
  }else{
    state.equippedItems[item.slot]=item.value;
    saveCloset();
    switchPage("homePage");
    setTimeout(()=>{spawnEffect(item.icon,8);showToast(item.message);render();},250);
    render();
  }
}

function switchRewardTab(tab){
  state.rewardTab=tab;
  render();
}
function handleCoupon(key){
  const item=couponItems[key];
  if(!item)return;
  if(state.coins<item.cost){
    const need=Math.max(0,item.cost-state.coins);
    showToast(item.name+" 교환까지 "+need+"코인 더 필요해요.");
    return;
  }
  state.coins-=item.cost;
  state.ownedCoupons[key]=Number(state.ownedCoupons[key]||0)+1;
  saveCoupons();
  render();
  showToast(item.message+" 혜택: "+item.benefit);
}


const actions={
  startFirstMapping:startFirstMapping,
  predictSoc:predictSocFromConditions,
  executeTopClean:executeTopClean,
  manualCleanAndGo:manualCleanAndGo,
  aiAutoClean:aiAutoClean,dirtyOnlyClean:dirtyOnlyClean,toggleNoGoMode:toggleNoGoMode,mapZone:handleMapZoneTap,
  selectHome:()=>selectScenario("home"),selectZone1:()=>selectScenario("zone",1),selectZone2:()=>selectScenario("zone",2),selectZone3:()=>selectScenario("zone",3),selectZone4:()=>selectScenario("zone",4),selectZone5:()=>selectScenario("zone",5),selectZone6:()=>selectScenario("zone",6),selectZone7:()=>selectScenario("zone",7),selectZone8:()=>selectScenario("zone",8),
  pet:petRobot,feed:feedRobot,play:playRobot,train:trainRobot,photo:takePhoto,clean:startCleaning,charge:chargeRobot,status:showStatus,
  // 홈의 "청소 기록" 버튼은 실시간 케어 기록이 있는 부품 케어 탭으로 이동합니다.
  record:()=>switchPage("batteryPage"),care:()=>switchPage("batteryPage"),event:()=>switchPage("eventPage"),decorate:decorateRobot,shop:()=>switchPage("rewardPage"),chargeFromBattery:()=>{switchPage("homePage");setTimeout(chargeRobot,220)},buyFood:buyFood,
  itemRibbon:()=>handleRewardItem("ribbon"),itemHat:()=>handleRewardItem("hat"),itemBunny:()=>handleRewardItem("bunny"),itemCat:()=>handleRewardItem("cat"),itemSparkle:()=>handleRewardItem("sparkle"),
  rewardTabItems:()=>switchRewardTab("items"),rewardTabCoupons:()=>switchRewardTab("coupons"),
  couponLg5:()=>handleCoupon("lg5"),couponCleanKit:()=>handleCoupon("cleanKit"),couponBatteryCare:()=>handleCoupon("batteryCare"),couponMoveIn:()=>handleCoupon("moveIn"),
  ribbon:()=>handleRewardItem("ribbon"),sparkle:()=>handleRewardItem("sparkle"),hat:()=>handleRewardItem("hat"),
  // ---- 2번째 탭: 부품 케어 ----
  partDetail:openPartDetail,
  // ---- 3번째 탭: 예약 청소 ----
  toggleCommute:toggleCommute,toggleDay:toggleDay,commuteMode:setCommuteMode,toggleTheme:toggleTheme,
  // ---- 4번째 탭: 이벤트 ----
  evTabFound:()=>switchEventTab("found"),evTabMission:()=>switchEventTab("mission"),evTabPhoto:()=>switchEventTab("photo"),
  foundItem:openFoundItem,foundMapBig:openFoundMapBig,claimMission:claimMission,photoOpen:openPhoto
};

document.addEventListener("click",(event)=>{const nav=event.target.closest("[data-page]");if(nav){switchPage(nav.dataset.page);return}const action=event.target.closest("[data-action]");if(action&&typeof actions[action.dataset.action]==="function"){actions[action.dataset.action](action,event)}});
// 1회차 학습 청소는 핵심 CTA라서, 이벤트 위임/터치/겹침 이슈가 있어도 반드시 동작하도록 여러 경로로 직접 연결합니다.
let lastLearnClickAt=0;
function triggerLearnButton(event){
  if(event){
    event.preventDefault();
    event.stopPropagation();
    if(event.stopImmediatePropagation)event.stopImmediatePropagation();
  }
  const now=Date.now();
  if(now-lastLearnClickAt<700)return;
  lastLearnClickAt=now;
  if(typeof startFirstMapping==="function")startFirstMapping();
}
window.__forceStartFirstMapping=triggerLearnButton;
const learnBtnDirect=$("learnBtn");
if(learnBtnDirect){
  learnBtnDirect.onclick=triggerLearnButton;
  learnBtnDirect.onpointerdown=triggerLearnButton;
  learnBtnDirect.onmousedown=triggerLearnButton;
  learnBtnDirect.ontouchstart=triggerLearnButton;
  learnBtnDirect.addEventListener("click",triggerLearnButton,true);
  learnBtnDirect.addEventListener("pointerdown",triggerLearnButton,true);
  learnBtnDirect.addEventListener("pointerup",triggerLearnButton,true);
  learnBtnDirect.addEventListener("mousedown",triggerLearnButton,true);
  learnBtnDirect.addEventListener("touchstart",triggerLearnButton,{capture:true,passive:false});
  learnBtnDirect.addEventListener("touchend",triggerLearnButton,{capture:true,passive:false});
}
document.addEventListener("pointerdown",(event)=>{
  const btn=event.target && event.target.closest ? event.target.closest("#learnBtn") : null;
  if(btn)triggerLearnButton(event);
},true);
document.addEventListener("touchstart",(event)=>{
  const target=event.target;
  const btn=target && target.closest ? target.closest("#learnBtn") : null;
  if(btn)triggerLearnButton(event);
},{capture:true,passive:false});

$("modalCancel").addEventListener("click",closeModal);$("modalConfirm").addEventListener("click",()=>modalConfirmHandler());$("modal").addEventListener("click",(event)=>{if(event.target===$("modal"))closeModal()});
["scopeSelect","cleanModeSelect","intensitySelect","todayStateSelect"].forEach(id=>{
  const el=$(id);
  if(el)el.addEventListener("change",()=>{
    if(state.profileReady && state.predicted){
      state.predicted=false;
      const loading=$('predictLoading');
      if(loading)loading.textContent="조건이 바뀌었어요. 오늘 청소 준비하기를 다시 눌러주세요.";
      render();
    }
  });
});
["leaveTime","returnTime"].forEach(id=>{const el=$(id);if(el)el.addEventListener("change",()=>render());});

setInterval(()=>{if(state.cleaning||state.charging||state.celebrating)return;const robot=$("robot");robot.classList.remove("look-left","look-right");const d=Math.random();if(d<.33)robot.classList.add("look-left");else if(d<.66)robot.classList.add("look-right");setTimeout(()=>robot.classList.remove("look-left","look-right"),1100)},2800);

fillTimeSelects();
populateConditionSelectors();
renderFound();
renderPhotos();
render();
state.notifiedClaimable=claimableCount();
</script>
</body>
</html>
"""

APP_HTML = APP_HTML.replace("__UI_PREDICTION_DATA__", UI_PREDICTION_JSON)
APP_HTML = APP_HTML.replace("__UI_MEDIA_DATA__", UI_MEDIA_JSON)

components.html(APP_HTML, height=1010, scrolling=False)
