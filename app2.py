import os
import cv2
import tempfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from fpdf import FPDF

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Audiogram Analyzer Pro", page_icon="🏥", layout="wide")

# --- LOAD MODEL (CACHED & PATCHED) ---
@st.cache_resource
def load_trained_model():
    _original_dense_init = tf.keras.layers.Dense.__init__
    def _patched_dense_init(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        return _original_dense_init(self, *args, **kwargs)
    
    tf.keras.layers.Dense.__init__ = _patched_dense_init
    model = tf.keras.models.load_model("Model_100K_FINAL_SNIPER_H.keras", safe_mode=False, compile=False)
    tf.keras.layers.Dense.__init__ = _original_dense_init
    return model

try:
    model = load_trained_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Model yükleme hatası: {e}")

# --- CLINICAL LOSS CATEGORIZER ---
def get_loss_category(db_value):
    if db_value <= 25.99: return "Normal (No Loss)"
    elif 26 <= db_value <= 40.99: return "Mild Hearing Loss"
    elif 41 <= db_value <= 55.99: return "Moderate Loss"
    elif 56 <= db_value <= 70.99: return "Moderately Severe"
    elif 71 <= db_value <= 90.99: return "Severe Loss"
    else: return "Profound Loss"

# --- DIGITAL PRISM: COLOR SPLITTER ---
def split_audiogram_colors(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Kırmızı Maske (Sağ Kulak)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    # Mavi Maske (Sol Kulak)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Sağ Kulak için Mavileri sil
    img_right = img.copy()
    img_right[mask_blue > 0] = [255, 255, 255] 

    # Sol Kulak için Kırmızıları sil
    img_left = img.copy()
    img_left[mask_red > 0] = [255, 255, 255]

    img_right_rgb = Image.fromarray(cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB))
    img_left_rgb = Image.fromarray(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))

    return img_right_rgb, img_left_rgb

# --- PDF GENERATOR (DUAL EAR SUPPORT) ---
import os
import tempfile
from fpdf import FPDF

def generate_pdf(patient_name, dict_r, dict_l):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt="CLINICAL AUDIOGRAM REPORT", ln=True, align='C')
    pdf.set_font("Arial", 'I', 10)
    pdf.cell(0, 8, txt="AI-Powered Digital Twin Analysis", ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"Patient Name: {patient_name}", ln=True)
    pdf.ln(5)
    
    # Resimleri Yan Yana Koyma
    tmp_paths = []
    y_before_img = pdf.get_y()
    
    if dict_r:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_r:
            dict_r['img_pil'].save(tmp_r.name)
            tmp_paths.append(tmp_r.name)
            pdf.image(tmp_r.name, x=15, y=y_before_img, w=85)
            
    if dict_l:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_l:
            dict_l['img_pil'].save(tmp_l.name)
            tmp_paths.append(tmp_l.name)
            x_pos = 110 if dict_r else 55
            pdf.image(tmp_l.name, x=x_pos, y=y_before_img, w=85)
            
    pdf.ln(90) # Resimlerin boyutu kadar aşağı in
    for p in tmp_paths: os.remove(p) # Çöpleri temizle
    
    # 1. Global Metrics
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="1. Global Metrics & Diagnosis", ln=True)
    pdf.set_font("Arial", '', 10)
    
    if dict_r and dict_l:
        pdf.cell(0, 8, txt=f"RIGHT EAR - PTA: {dict_r['pta']:.1f} dB ({dict_r['who']}) | Disability: {dict_r['disability']:.1f}%", ln=True)
        pdf.cell(0, 8, txt=f"LEFT EAR - PTA: {dict_l['pta']:.1f} dB ({dict_l['who']}) | Disability: {dict_l['disability']:.1f}%", ln=True)
    elif dict_r:
        pdf.cell(0, 8, txt=f"RIGHT EAR - PTA: {dict_r['pta']:.1f} dB ({dict_r['who']}) | Disability: {dict_r['disability']:.1f}%", ln=True)
    elif dict_l:
        pdf.cell(0, 8, txt=f"LEFT EAR - PTA: {dict_l['pta']:.1f} dB ({dict_l['who']}) | Disability: {dict_l['disability']:.1f}%", ln=True)
    pdf.ln(5)
    
    # 2. Detailed Frequency Logs
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="2. Detailed Frequency Thresholds", ln=True)
    pdf.set_font("Arial", 'B', 9)
    
    if dict_r and dict_l:
        # Yan Yana Tablo Başlıkları
        pdf.cell(20, 8, "Freq", border=1, align='C')
        pdf.cell(20, 8, "R-dB", border=1, align='C')
        pdf.cell(50, 8, "Right Status", border=1, align='C')
        pdf.cell(5, 8, "", border=0) # Boşluk
        pdf.cell(20, 8, "Freq", border=1, align='C')
        pdf.cell(20, 8, "L-dB", border=1, align='C')
        pdf.cell(50, 8, "Left Status", border=1, ln=True, align='C')
        
        pdf.set_font("Arial", '', 9)
        for i in range(len(dict_r['freqs'])):
            f = dict_r['freqs'][i]
            # Sağ veriler
            pdf.cell(20, 8, f"{f} Hz", border=1, align='C')
            pdf.cell(20, 8, f"{dict_r['preds'][i]:.1f}", border=1, align='C')
            pdf.cell(50, 8, get_loss_category(dict_r['preds'][i]), border=1, align='C')
            pdf.cell(5, 8, "", border=0) # Boşluk
            # Sol veriler
            pdf.cell(20, 8, f"{f} Hz", border=1, align='C')
            pdf.cell(20, 8, f"{dict_l['preds'][i]:.1f}", border=1, align='C')
            pdf.cell(50, 8, get_loss_category(dict_l['preds'][i]), border=1, ln=True, align='C')
            
    else:
        # Tek Kulak Tablosu
        active_dict = dict_r if dict_r else dict_l
        pdf.cell(40, 8, "Frequency", border=1, align='C')
        pdf.cell(40, 8, "Threshold (dB)", border=1, align='C')
        pdf.cell(80, 8, "Clinical Status", border=1, ln=True, align='C')
        pdf.set_font("Arial", '', 9)
        for i in range(len(active_dict['freqs'])):
            f = active_dict['freqs'][i]
            pdf.cell(40, 8, f"{f} Hz", border=1, align='C')
            pdf.cell(40, 8, f"{active_dict['preds'][i]:.1f}", border=1, align='C')
            pdf.cell(80, 8, get_loss_category(active_dict['preds'][i]), border=1, ln=True, align='C')

    # =================================================================================
    # 3. DIAGNOSIS: MORPHOLOGY (YENİ EKLENEN KLİNİK ANALİZ BÖLÜMÜ)
    # =================================================================================
    # =================================================================================
    # 3. DIAGNOSIS: MORPHOLOGY (ARAYÜZ METİNLERİ VE HASSASİYETİ İLE %100 EŞLEŞTİRİLDİ)
    # =================================================================================
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="3. DIAGNOSIS: MORPHOLOGY", ln=True)
    pdf.set_font("Arial", '', 10)
    
    uyarilar = []
    
    # Sağ Kulak Analizi
    if dict_r:
        freqs_list_r = list(dict_r['freqs'])
        if 2000 in freqs_list_r and 4000 in freqs_list_r and 8000 in freqs_list_r:
            r_2k = dict_r['preds'][freqs_list_r.index(2000)]
            r_4k = dict_r['preds'][freqs_list_r.index(4000)]
            r_8k = dict_r['preds'][freqs_list_r.index(8000)]
            
            # Kural: 2000'den 4000'e en az 15 dB düşüş VE 4000'den 8000'e en az 10 dB toparlanma
            if (r_4k >= r_2k + 15) and (r_4k >= r_8k + 10):
                uyarilar.append(f"RIGHT EAR -> DIAGNOSIS: CLASSIC V-NOTCH\n"
                                f"A sudden drop at 4000Hz ({r_4k:.1f} dB) followed by a recovery at 8000Hz ({r_8k:.1f} dB) was detected.\n"
                                f"CONCLUSION: High probability of Noise-Induced Hearing Loss (NIHL) or Tinnitus.")
            elif (r_4k >= r_2k + 15):
                uyarilar.append(f"RIGHT EAR -> DIAGNOSIS: SLOPING LOSS\n"
                                f"High-frequency drop detected without significant recovery at 8000Hz.")
            else:
                uyarilar.append("RIGHT EAR -> DIAGNOSIS: NORMAL MORPHOLOGY\n"
                                "The curve morphology does not exhibit any severe acoustic notch.")

    # Sol Kulak Analizi
    if dict_l:
        freqs_list_l = list(dict_l['freqs'])
        if 2000 in freqs_list_l and 4000 in freqs_list_l and 8000 in freqs_list_l:
            l_2k = dict_l['preds'][freqs_list_l.index(2000)]
            l_4k = dict_l['preds'][freqs_list_l.index(4000)]
            l_8k = dict_l['preds'][freqs_list_l.index(8000)]
            
            if (l_4k >= l_2k + 15) and (l_4k >= l_8k + 10):
                uyarilar.append(f"LEFT EAR -> DIAGNOSIS: CLASSIC V-NOTCH\n"
                                f"A sudden drop at 4000Hz ({l_4k:.1f} dB) followed by a recovery at 8000Hz ({l_8k:.1f} dB) was detected.\n"
                                f"CONCLUSION: High probability of Noise-Induced Hearing Loss (NIHL) or Tinnitus.")
            elif (l_4k >= l_2k + 15):
                uyarilar.append(f"LEFT EAR -> DIAGNOSIS: SLOPING LOSS\n"
                                f"High-frequency drop detected without significant recovery at 8000Hz.")
            else:
                uyarilar.append("LEFT EAR -> DIAGNOSIS: NORMAL MORPHOLOGY\n"
                                "The curve morphology does not exhibit any severe acoustic notch.")
            
    # Genel Asimetri Kontrolü
    if dict_r and dict_l:
        delta = abs(dict_r['pta'] - dict_l['pta'])
        if delta > 10:
            uyarilar.append(f"GENERAL WARNING -> Clinical asymmetry detected between ears (Delta: {delta:.1f} dB HL)")
            
    # Uyarıları gri kutucuklar (Alert Box) içinde PDF'e basma
    for uyari in uyarilar:
        pdf.set_fill_color(245, 245, 245) 
        # Multi_cell kullanımı sayesinde satır atlamaları (\n) kutu içinde kusursuz görünür
        pdf.multi_cell(0, 6, uyari, border=1, fill=True) 
        pdf.ln(3) # Kutular arası boşluk
    # =================================================================================
    # =================================================================================
    # =================================================================================
    # =================================================================================

    # Alt Bilgi (Footer)
    pdf.ln(15)
    pdf.set_font("Arial", 'I', 9)
    pdf.set_text_color(150, 150, 150)
    pdf.multi_cell(0, 5, "* This report is automatically generated by AI. Please consult an ENT specialist for final medical diagnosis.")
    
    return pdf.output(dest="S").encode("latin1")


# --- HELPER FUNCTION: RENDER EXACT UI ---
def render_analysis_ui(img_pil, ear_side, line_color, marker_style):
    img_array = np.array(img_pil.resize((224, 224)))
    img_input = np.expand_dims(img_array / 255.0, axis=0)

    # Inference
    predictions = model.predict(img_input)[0]
    db_250, db_500, db_1000, db_2000, db_4000, db_8000 = predictions[0], predictions[1], predictions[2], predictions[3], predictions[4], predictions[5]
    
    freqs = np.array([250, 500, 1000, 2000, 4000, 8000])
    preds = np.array([db_250, db_500, db_1000, db_2000, db_4000, db_8000])

    # Analytics
    pta_average = (db_500 + db_1000 + db_2000 + db_4000) / 4
    has_tinnitus = (db_4000 - db_2000) > 15 and (db_4000 - db_8000) > 10 and db_4000 > 25
    disability_rate = (pta_average - 25) * 1.5 if pta_average > 25 else 0
    disability_rate = min(disability_rate, 100)
    who_class = get_loss_category(pta_average)

    # --- TOP ROW ---
    st.subheader(f"📊 {ear_side} - Threshold Detection")
    cols = st.columns(7)
    cols[0].metric("250 Hz", f"{db_250:.1f} dB")
    cols[1].metric("500 Hz", f"{db_500:.1f} dB")
    cols[2].metric("1000 Hz", f"{db_1000:.1f} dB")
    cols[3].metric("2000 Hz", f"{db_2000:.1f} dB")
    cols[4].metric("4000 Hz", f"{db_4000:.1f} dB")
    cols[5].metric("8000 Hz", f"{db_8000:.1f} dB")
    cols[6].metric("PTA (Avg)", f"{pta_average:.1f} dB", who_class, delta_color="inverse")
    st.markdown("---")

    # --- MAIN CONTENT ---
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📸 Source & Digital Twin")
        st.image(img_pil, caption=f"Analyzed Side: {ear_side}", width=350)
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("### 🎯 Interactive Target Marker")
        custom_marker_freq = st.slider(
            f"Ara frekansları ({ear_side}) incelemek için sürgüyü kaydırın:", 
            min_value=250, max_value=8000, value=2493, step=1, key=f"slider_{ear_side}"
        )
        
        log_freqs = np.log10(freqs)
        log_custom = np.log10(custom_marker_freq)
        custom_marker_db = np.interp(log_custom, log_freqs, preds)
        custom_loss = max(0, custom_marker_db - normal_threshold)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(freqs, preds, marker=marker_style, color=line_color, linewidth=2.5, markersize=8, label="AI Prediction")
        ax.axhline(y=normal_threshold, color='green', linestyle='--', linewidth=2, label=f"Normal Limit ({normal_threshold} dB)")
        ax.axhspan(-10, normal_threshold, facecolor='green', alpha=0.1)
        
        ax.plot(custom_marker_freq, custom_marker_db, marker='*', color='purple', markersize=18, label=f"Marker ({custom_marker_freq}Hz: {custom_marker_db:.1f}dB)")
        ax.vlines(x=custom_marker_freq, ymin=-10, ymax=custom_marker_db, color='purple', linestyle=':', linewidth=2)

        ax.set_xscale('log')
        ax.set_xticks(freqs)
        ax.set_xticklabels(freqs)
        ax.set_ylim(120, -10)
        ax.grid(True, which="both", linestyle='--', alpha=0.5)
        ax.set_title(f"AI Generated Audiogram Trace ({ear_side})", fontsize=11)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Hearing Level (dB HL)")
        ax.legend()
        st.pyplot(fig)

    with col2:
        st.subheader("📜 Comprehensive Medical Report")
        
        st.info(f"**👤 PATIENT PROFILE & GLOBAL METRICS**\n\n"
                f"**Analyzed Side:** {ear_side}\n\n"
                f"**PTA (500-4000 Hz):** {pta_average:.1f} dB HL\n\n"
                f"**Severity Status:** {who_class}\n\n"
                f"**General Disability:** {disability_rate:.1f}%")

        with st.container(border=True):
            st.markdown("#### 📊 Detailed Frequency Log")
            st.markdown(f"- **[ 250 Hz ]:** {db_250:.1f} dB ➔ *{get_loss_category(db_250)}*")
            st.markdown(f"- **[ 500 Hz ]:** {db_500:.1f} dB ➔ *{get_loss_category(db_500)}*")
            st.markdown(f"- **[ 1000 Hz ]:** {db_1000:.1f} dB ➔ *{get_loss_category(db_1000)}*")
            st.markdown(f"- **[ 2000 Hz ]:** {db_2000:.1f} dB ➔ *{get_loss_category(db_2000)}*")
            st.markdown(f"- **[ 4000 Hz ]:** {db_4000:.1f} dB ➔ *{get_loss_category(db_4000)}*")
            st.markdown(f"- **[ 8000 Hz ]:** {db_8000:.1f} dB ➔ *{get_loss_category(db_8000)}*")

        with st.container(border=True):
            st.markdown("#### 🎯 Interactive Target Analytics")
            st.write(f"**Selected Frequency:** {custom_marker_freq} Hz")
            st.write(f"**AI Interpolated Value:** {custom_marker_db:.1f} dB HL")
            if custom_loss > 0:
                st.warning(f"**Deviation:** {custom_loss:.1f} dB loss detected compared to healthy threshold.")
            else:
                st.success(f"**Deviation:** Hearing is strictly within healthy limits at this target.")

        if has_tinnitus:
            st.error(f"**🚨 DIAGNOSIS: CLASSIC V-NOTCH**\n\n"
                     f"A sudden drop at 4000Hz (**{db_4000:.1f} dB**) followed by a recovery at 8000Hz (**{db_8000:.1f} dB**) was detected.\n\n"
                     f"**📌 CONCLUSION:** High probability of Noise-Induced Hearing Loss (NIHL) or Tinnitus.")
        else:
            st.success(f"**✅ DIAGNOSIS: MORPHOLOGY**\n\n"
                       f"The curve morphology does not exhibit any severe acoustic notch.")
            
    # PDF İçin Verileri Geri Döndür
    return {
        "img_pil": img_pil, "ear_side": ear_side, "pta": pta_average, 
        "who": who_class, "disability": disability_rate, "freqs": freqs, "preds": preds
    }


# --- GUI HEADER ---
st.title("🏥 AI Clinical Audiogram Analysis System (V2)")
st.markdown("*Advanced Deep Learning Diagnostic Support Tool (6-Frequency Full Spectrum)*")
st.markdown("---")

# --- SIDEBAR ---
st.sidebar.header("📁 Patient Record")
patient_name = st.sidebar.text_input("Patient Name", "Bilinmeyen Hasta")
normal_threshold = 20

st.sidebar.header("📤 Upload Options")
upload_mode = st.sidebar.radio("Mode Seçimi:", ["Tek Birleşik Fotoğraf (Sağ + Sol)", "Ayrı Ayrı Fotoğraflar"])

img_r, img_l = None, None

if upload_mode == "Tek Birleşik Fotoğraf (Sağ + Sol)":
    uploaded_file = st.sidebar.file_uploader("Birleşik Odyogram Yükle", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.sidebar.success("✨ Kırmızı ve Mavi Çizgiler Ayrıştırılıyor...")
        img_r, img_l = split_audiogram_colors(uploaded_file)
else:
    up_r = st.sidebar.file_uploader("🔴 SAĞ Kulak Yükle (Kırmızı)", type=["jpg", "png", "jpeg"])
    up_l = st.sidebar.file_uploader("🔵 SOL Kulak Yükle (Mavi)", type=["jpg", "png", "jpeg"])
    if up_r: img_r = Image.open(up_r).convert('RGB')
    if up_l: img_l = Image.open(up_l).convert('RGB')

# --- MAIN RENDER LOGIC ---
if (img_r or img_l) and model_loaded:
    dict_r, dict_l = None, None
    
    # Görünümleri alt alta aynen basıyoruz
    if img_r:
        dict_r = render_analysis_ui(img_r, "RIGHT EAR", 'red', 'o')
        st.markdown("<br><br>", unsafe_allow_html=True)
        
    if img_l:
        dict_l = render_analysis_ui(img_l, "LEFT EAR", 'blue', 'x')
        st.markdown("<br><br>", unsafe_allow_html=True)

    # --- PDF EXPORT ---
    st.markdown("---")
    st.subheader("📄 Export Results")
    
    pdf_bytes = generate_pdf(patient_name, dict_r, dict_l)
    
    st.download_button(
        label="📥 Download Clinical PDF Report",
        data=pdf_bytes,
        file_name=f"Audiology_Report_{patient_name.replace(' ', '_')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )
else:
    st.info("👈 Lütfen analizi başlatmak için sol menüden odyogram resmi yükleyin.")