import streamlit as st
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import random
import csv
import os
import time
from datetime import datetime
from transformers import pipeline

# --- 1. KONFIGURACJA ---
DATASET_PATH = 'dataset.csv'
MODEL_NAME = "Panda0116/emotion-classification-model"

st.set_page_config(page_title="Eksperyment Muzyczny", layout="centered")

# Mapowanie wynikow modelu na nazwy emocji
MODEL_OUTPUT_MAPPING = {
    "LABEL_0": "Sadness",
    "LABEL_1": "Happiness",
    "LABEL_2": "Tenderness",
    "LABEL_3": "Anger",
    "LABEL_4": "Fear"
}

# --- 2. LADOWANIE ZASOBOW (CACHE) ---
@st.cache_resource
def load_resources():
    print("\n[SYSTEM] Ladowanie zasobow...")
    # A. Dataset
    try:
        df = pd.read_csv(DATASET_PATH)
        df.columns = [col.lower() for col in df.columns] 
    except FileNotFoundError:
        st.error(f"Brak pliku {DATASET_PATH}.")
        st.stop()

    # B. NLP
    classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None)

    # C. Fuzzy Logic
    emotion_cat = ctrl.Antecedent(np.arange(0, 5, 1), 'emotion_category')
    confidence = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'confidence')
    valence = ctrl.Consequent(np.arange(1, 10, 1), 'valence')
    arousal = ctrl.Consequent(np.arange(1, 10, 1), 'arousal')

    emotion_cat['Sadness'] = fuzz.gaussmf(emotion_cat.universe, 0, 0.5)
    emotion_cat['Happiness'] = fuzz.gaussmf(emotion_cat.universe, 1, 0.5)
    emotion_cat['Anger'] = fuzz.gaussmf(emotion_cat.universe, 2, 0.5)
    emotion_cat['Fear'] = fuzz.gaussmf(emotion_cat.universe, 3, 0.5)
    emotion_cat['Tenderness'] = fuzz.gaussmf(emotion_cat.universe, 4, 0.5)

    confidence['Low'] = fuzz.gaussmf(confidence.universe, 0, 0.15)
    confidence['Medium'] = fuzz.gaussmf(confidence.universe, 0.5, 0.15)
    confidence['High'] = fuzz.gaussmf(confidence.universe, 1, 0.15)

    valence.automf(5, names=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    arousal.automf(5, names=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

    # Definicja Regul
    rule1 = ctrl.Rule(emotion_cat['Sadness'] & confidence['High'], [valence['Very Low'], arousal['Very Low']])
    rule2 = ctrl.Rule(emotion_cat['Happiness'] & confidence['High'], [valence['Very High'], arousal['High']])
    rule3 = ctrl.Rule(emotion_cat['Anger'] & confidence['High'], [valence['Low'], arousal['Very High']])
    rule4 = ctrl.Rule(emotion_cat['Fear'] & confidence['High'], [valence['Low'], arousal['Very High']])
    rule5 = ctrl.Rule(emotion_cat['Tenderness'] & confidence['High'], [valence['High'], arousal['Low']])
    rule_low = ctrl.Rule(confidence['Low'], [valence['Medium'], arousal['Medium']])

    emotion_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule_low])
    fuzzy_sim = ctrl.ControlSystemSimulation(emotion_ctrl)
    
    print("[SYSTEM] Gotowe!\n")
    return df, classifier, fuzzy_sim

try:
    spotify_df, nlp_classifier, fuzzy_system = load_resources()
except Exception as e:
    st.error(f"Blad inicjalizacji: {e}")
    st.stop()


# --- 3. LOGIKA PRZETWARZANIA (Z LOGAMI) ---
def process_text_smart(text, classifier, fuzzy_sim, df):
    print("\n" + "="*50)
    print(f"[ALGO START] Tekst: \"{text[:50]}...\"")
    
    # 1. NLP
    preds = classifier(text)
    sorted_preds = sorted(preds[0], key=lambda x: x['score'], reverse=True)
    top_pred = sorted_preds[0]
    
    emotion_name = MODEL_OUTPUT_MAPPING.get(top_pred['label'], "Unknown")
    confidence_score = top_pred['score']

    print(f"[NLP] Emocja: {emotion_name.upper()} (Pewnosc: {confidence_score:.4f})")

    # 2. Fuzzy Logic
    emotion_map_idx = {"Sadness": 0, "Happiness": 1, "Anger": 2, "Fear": 3, "Tenderness": 4}
    emotion_idx = emotion_map_idx.get(emotion_name, 0)

    fuzzy_sim.input['emotion_category'] = emotion_idx
    fuzzy_sim.input['confidence'] = confidence_score
    
    try:
        fuzzy_sim.compute()
        val_fuzzy = fuzzy_sim.output['valence']
        aro_fuzzy = fuzzy_sim.output['arousal']
        print(f"[FUZZY] Wyliczono (1-9): Valence={val_fuzzy:.2f}, Arousal={aro_fuzzy:.2f}")
    except:
        val_fuzzy, aro_fuzzy = 5.0, 5.0
        print("[FUZZY] Blad obliczen, fallback 5.0")

    # 3. Norm
    target_valence = max(0.0, min(1.0, (val_fuzzy - 1) / 8.0))
    target_energy = max(0.0, min(1.0, (aro_fuzzy - 1) / 8.0))
    print(f"[NORM] Cel Spotify (0-1): Valence={target_valence:.2f}, Energy={target_energy:.2f}")

    # 4. Filter
    genres_to_search = []
    filtered_df = df.copy()
    
    print("[RULES] Dobieranie gatunku...")
    if target_energy >= 0.55:
        if target_valence < 0.45: 
            print(" -> High Energy + Negative (Gniew/Strach)")
            genres_to_search = ['industrial', 'dark-ambient', 'breakbeat', 'gabber', 'hard-techno', 'techno', 'minimal-techno', 'metal']
            if 'mode' in filtered_df.columns:
                print(" -> Filtr: MINOR KEY (Mrok)")
                filtered_df = filtered_df[filtered_df['mode'] == 0]
        else: 
            print(" -> High Energy + Positive (Radosc)")
            genres_to_search = ['house', 'trance', 'happy-hardcore', 'dance', 'edm', 'chicago-house']
            if 'mode' in filtered_df.columns:
                print(" -> Filtr: MAJOR KEY (Jasnosc)")
                filtered_df = filtered_df[filtered_df['mode'] == 1]
    else:
        if target_valence >= 0.50: 
            print(" -> Low Energy + Positive (Relaks)")
            genres_to_search = ['ambient', 'piano', 'new-age', 'jazz']
            if 'mode' in filtered_df.columns:
                print(" -> Filtr: MAJOR KEY (Anty-Burzum)")
                filtered_df = filtered_df[filtered_df['mode'] == 1]
        else: 
            print(" -> Low Energy + Negative (Smutek)")
            genres_to_search = ['ambient', 'classical']
            print(" -> Filtr: Brak")

    pattern = '|'.join(genres_to_search)
    candidates = filtered_df[filtered_df['track_genre'].str.contains(pattern, case=False, na=False)].copy()
    
    if 'instrumentalness' in candidates.columns:
        candidates = candidates[candidates['instrumentalness'] > 0.5]

    tolerance = 0.1
    result = pd.DataFrame()
    while result.empty and tolerance <= 0.6:
        result = candidates[
            (candidates['valence'].between(target_valence - tolerance, target_valence + tolerance)) &
            (candidates['energy'].between(target_energy - tolerance, target_energy + tolerance))
        ]
        tolerance += 0.05
    
    if result.empty:
        print("[FALLBACK] Brak idealnego dopasowania, biore losowy z gatunku.")
        if not candidates.empty:
            final = candidates.sample(1).iloc[0]
        else:
            final = df.sample(1).iloc[0]
    else:
        final = result.sample(1).iloc[0]

    print(f"[WYNIK] Wybrano: {final['track_name']} - {final['artists']}")
    print("="*50 + "\n")
    return final

def get_random_track(df):
    print("\n" + "="*50)
    print("[RANDOM] Losowanie utworu (Grupa Kontrolna)...")
    t = df.sample(1).iloc[0]
    print(f"[WYNIK] Wylosowano: {t['track_name']}")
    print("="*50 + "\n")
    return t


# --- 5. INTERFEJS UŻYTKOWNIKA ---

# GOTOWE SCENARIUSZE
PRESET_SCENARIOS = {
    "GNIEW (The Boscombe Valley Mystery)": 
    "His face was livid with fury, his eyes blazing, and his whole frame trembling with passion. 'You villain!' he screamed, clenching his fists until the knuckles were white. 'I will not listen to another word! You have betrayed me, you have ruined everything!' He raised his cane as if to strike, his voice choking with an overwhelming, uncontrollable rage that seemed to consume him entirely.",
    
    "STRACH (The Speckled Band)": 
    "'It is fear, Mr. Holmes. It is terror.' She raised her veil as she spoke, and we could see that she was indeed in a pitiable state of agitation, her face all drawn and grey, with restless, frightened eyes, like those of some hunted animal. Her features and figure were those of a woman of thirty, but her hair was shot with premature grey, and her expression was weary and haggard. She shuddered as she sat, and I could see the sweat beads upon her forehead.",
    
    "SMUTEK (The Beryl Coronet)": 
    "'I am a ruined man, Mr. Holmes—a ruined man! My honour is gone, my good name is lost, and I have to face my family with the knowledge that I have brought them to shame. I kept the treasure in my own hands, and it is gone. God help me! God help me!' He threw up his hands in an agony of despair and pressed his fingers into his hair, swaying backwards and forwards in his chair.",
    
    "RADOSC (A Study in Scarlet / Success)": 
    "A flush of triumph appeared upon his pale cheeks, and his eyes shone with the pure joy of the solution. 'We have done it, Watson!' he cried, clapping his hands together with a boyish laugh of delight. 'The case is clear! The mystery is solved! It is a masterpiece!' His face radiated happiness and excitement, the heavy burden of the investigation lifting to reveal a spirit light and jubilant.",
    
    "CZULOSC (The Man with the Twisted Lip)": 
    "The woman rushed forward and threw her arms round the man's neck. 'Oh, Neville!' she cried, 'I knew that you were safe! I knew that you were safe!' She held him close with a devotion that spoke more than words, while he pressed his cheek against hers, stroking her hair with a gentle, trembling hand. In that moment of reunion, the grim surroundings of the opium den seemed to fade away."
}

def main():
    # --- PASEK BOCZNY ---
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Spotify_logo_without_text.svg/768px-Spotify_logo_without_text.svg.png", width=50)
        st.title("Panel Badawczy")
        
        if 'session_results' not in st.session_state:
            st.session_state.session_results = []
            
        count = len(st.session_state.session_results)
        st.metric("Liczba testow", count)
        st.progress(min(count / 10, 1.0))
        
        st.markdown("---")
        st.markdown("###Instrukcja dla testera:")
        st.info("""
        1. Kliknij **START**.
        2. **Najpierw włącz muzykę** (Play na Spotify).
        3. Czytaj tekst słuchając muzyki.
        4. Oceń, czy utwór buduje odpowiedni nastrój.
        5. Wykonaj min. **5 prób**.
        6. Pobierz/Skopiuj wyniki.
        """)
        
        if count > 0:
            st.markdown("---")
            st.subheader("Opcja 1: Komputer")
            df_res = pd.DataFrame(st.session_state.session_results)
            csv_data = df_res.to_csv(index=False).encode('utf-8')
            timestamp = datetime.now().strftime("%d-%m_%H-%M")
            
            st.download_button(
                label="📥 POBIERZ WYNIKI (CSV)",
                data=csv_data,
                file_name=f"wyniki_{timestamp}.csv",
                mime='text/csv',
                type="primary"
            )
            
            st.markdown("---")
            st.subheader("Opcja 2: Telefon")
            st.caption("Kliknij ikonkę kopiowania w rogu i wyślij mi to na Messengerze:")
            
            # Generowanie tekstu CSV do skopiowania
            csv_text = df_res.to_csv(index=False)
            st.code(csv_text, language='csv')
            
        else:
            st.warning("Zrob test, aby zobaczyc opcje pobierania.")

    # --- GLOWNE OKNO ---
    st.title("AI Music Experiment")
    st.markdown("""
    **Ślepy test do pracy dyplomowej.**
    Celem jest sprawdzenie, czy AI potrafi dobrać muzykę do odpowiednich emocji w tekście, która sprawi, że czytanie będzie przyjemniejsze i bardziej immersyjne.
    """)
    st.markdown("---")

    # Inicjalizacja Stanow
    if 'phase' not in st.session_state: st.session_state.phase = 'INPUT'
    if 'rec_method' not in st.session_state: st.session_state.rec_method = None
    if 'track' not in st.session_state: st.session_state.track = None
    if 'user_text' not in st.session_state: st.session_state.user_text = ""
    if 'scenario_name' not in st.session_state: st.session_state.scenario_name = ""

    # FAZA 1: WEJSCIE
    if st.session_state.phase == 'INPUT':
        st.subheader("1. Rozpocznij test")
        st.info("Kliknij przycisk poniżej. System wylosuje scenę z książki i dobierze tło muzyczne.")

        if st.button("Wylosuj Tekst i Muzykę (START)", type="primary", use_container_width=True):
            # 1. Losowanie tekstu
            scenario_name, scenario_text = random.choice(list(PRESET_SCENARIOS.items()))
            st.session_state.user_text = scenario_text
            st.session_state.scenario_name = scenario_name
            
            # 2. Losowanie metody (Blind Test 50/50)
            if random.random() < 0.5:
                st.session_state.rec_method = 'ALGO'
                with st.spinner("AI analizuje emocje w tekście..."):
                    time.sleep(0.5)
                    st.session_state.track = process_text_smart(
                        scenario_text, nlp_classifier, fuzzy_system, spotify_df
                    )
            else:
                st.session_state.rec_method = 'RANDOM'
                with st.spinner("Szukanie utworu w bazie..."):
                    time.sleep(0.5)
                    st.session_state.track = get_random_track(spotify_df)
            
            st.session_state.phase = 'RATING'
            st.rerun()

    # FAZA 2: OCENA
    elif st.session_state.phase == 'RATING':
        st.subheader("2. Odsłuch i Czytanie")
        
        # Player
        track = st.session_state.track
        st.markdown("###Krok 1: Włącz muzykę")
        embed_url = f"https://open.spotify.com/embed/track/{track['track_id']}?utm_source=generator"
        st.components.v1.iframe(embed_url, height=152)
        
        st.markdown("###Krok 2: Przeczytaj tekst")
        st.caption(f"Scena: {st.session_state.scenario_name}")
        
        # Ładniejsza prezentacja tekstu (jako cytat)
        st.markdown(f"""
        > *"{st.session_state.user_text}"*
        """)
        
        st.markdown("---")
        st.markdown("###Krok 3: Oceń wrażenia")
        st.write("Czy ta muzyka pasuje do emocji w tekście? Czy pomaga się wczuć?")
        
        with st.form("rating"):
            rating = st.slider("Ocena dopasowania", 1, 10, 5, help="1 = Zupełnie nie pasuje, 10 = Idealny klimat")
            comment = st.text_input("Twój komentarz (opcjonalny)")
            
            if st.form_submit_button("Zatwierdź Ocenę", type="primary"):
                res = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'scenario': st.session_state.scenario_name,
                    'user_text': st.session_state.user_text,
                    'method': st.session_state.rec_method,
                    'rating': rating,
                    'comment': comment,
                    'track_name': track['track_name'],
                    'artist': track['artists'],
                    'genre': track['track_genre'],
                    'valence': track['valence'],
                    'energy': track['energy'],
                    'mode': track.get('mode', 'N/A')
                }
                st.session_state.session_results.append(res)
                st.session_state.phase = 'REVEAL'
                st.rerun()

    # FAZA 3: WYNIK
    elif st.session_state.phase == 'REVEAL':
        st.subheader("3. Wynik")
        
        if st.session_state.rec_method == 'ALGO':
            st.success("To był **ALGORYTM AI**")
        else:
            st.warning("To był **LOSOWY UTWÓR**")
            
        st.write(f"Utwór: **{st.session_state.track['track_name']}** - {st.session_state.track['artists']}")
        st.caption(f"Gatunek: {st.session_state.track['track_genre']}")
        
        col1, col2 = st.columns(2)
        with col1:
             if st.button("Kolejny Test", type="primary", use_container_width=True):
                st.session_state.phase = 'INPUT'
                st.session_state.user_text = ""
                st.rerun()
        with col2:
            st.info("Pamiętaj o pobraniu wyników po zakończeniu!")

if __name__ == "__main__":
    main()