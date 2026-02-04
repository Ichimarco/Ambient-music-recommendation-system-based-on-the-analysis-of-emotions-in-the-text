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
    print("\n[SYSTEM] Inicjalizacja... Ladowanie modelu i danych...")
    # A. Dataset
    try:
        df = pd.read_csv(DATASET_PATH)
        df.columns = [col.lower() for col in df.columns] 
        print(f"[SYSTEM] Dataset zaladowany: {len(df)} utworow.")
    except FileNotFoundError:
        st.error(f"Brak pliku {DATASET_PATH}.")
        st.stop()

    # B. NLP
    classifier = pipeline("text-classification", model=MODEL_NAME, top_k=None)
    print("[SYSTEM] Model NLP zaladowany.")

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
    
    print("[SYSTEM] Logika rozmyta gotowa. Aplikacja startuje!\n")
    return df, classifier, fuzzy_sim

try:
    spotify_df, nlp_classifier, fuzzy_system = load_resources()
except Exception as e:
    st.error(f"Blad inicjalizacji: {e}")
    st.stop()


# --- 3. LOGIKA PRZETWARZANIA ---
def process_text_smart(text, classifier, fuzzy_sim, df):
    print("\n" + "="*50)
    print(f"[ALGO START] Analiza tekstu: \"{text[:40]}...\"")
    
    # 1. NLP
    preds = classifier(text)
    sorted_preds = sorted(preds[0], key=lambda x: x['score'], reverse=True)
    top_pred = sorted_preds[0]
    emotion_name = MODEL_OUTPUT_MAPPING.get(top_pred['label'], "Unknown")
    confidence_score = top_pred['score']
    
    print(f"[NLP] Wykryta emocja: {emotion_name.upper()} (Pewnosc: {confidence_score:.4f})")

    # 2. Fuzzy Logic
    emotion_map_idx = {"Sadness": 0, "Happiness": 1, "Anger": 2, "Fear": 3, "Tenderness": 4}
    emotion_idx = emotion_map_idx.get(emotion_name, 0)

    fuzzy_sim.input['emotion_category'] = emotion_idx
    fuzzy_sim.input['confidence'] = confidence_score
    
    try:
        fuzzy_sim.compute()
        val_fuzzy = fuzzy_sim.output['valence']
        aro_fuzzy = fuzzy_sim.output['arousal']
        print(f"[FUZZY] Wynik (1-10): Valence={val_fuzzy:.2f}, Arousal={aro_fuzzy:.2f}")
    except:
        val_fuzzy, aro_fuzzy = 5.0, 5.0
        print("[FUZZY] Blad obliczen, uzywam wartosci domyslnych 5.0")

    # 3. Norm & Override (Safety Net)
    target_valence = max(0.0, min(1.0, (val_fuzzy - 1) / 8.0))
    target_energy = max(0.0, min(1.0, (aro_fuzzy - 1) / 8.0))

    override = False
    if emotion_name == "Tenderness":
        target_valence, target_energy = 0.7, 0.2
        override = True
    elif emotion_name == "Fear":
        target_valence, target_energy = 0.2, 0.7
        override = True
    elif emotion_name == "Anger":
        target_valence, target_energy = 0.1, 0.9
        override = True
    elif emotion_name == "Happiness" and target_energy < 0.5:
        target_valence, target_energy = 0.9, 0.8
        override = True
    elif emotion_name == "Sadness" and target_energy > 0.4:
        target_valence, target_energy = 0.1, 0.2
        override = True
    
    if override:
        print(f"[RULES] Zastosowano Safety Net dla emocji {emotion_name}.")

    print(f"[TARGET] Cel Spotify: Valence={target_valence:.2f}, Energy={target_energy:.2f}")

    # 4. Filter
    genres_to_search = []
    filtered_df = df.copy()
    
    if emotion_name == "Tenderness":
        genres_to_search = ['piano', 'ambient', 'new-age', 'romantic', 'guitar', 'chill']
        if 'mode' in filtered_df.columns: filtered_df = filtered_df[filtered_df['mode'] == 1]
    elif target_energy >= 0.55:
        if target_valence < 0.45: 
            genres_to_search = ['industrial', 'dark-ambient', 'breakbeat', 'gabber', 'hard-techno', 'techno', 'minimal-techno', 'metal']
            if 'mode' in filtered_df.columns: filtered_df = filtered_df[filtered_df['mode'] == 0]
        else: 
            genres_to_search = ['house', 'trance', 'happy-hardcore', 'dance', 'edm', 'chicago-house']
            if 'mode' in filtered_df.columns: filtered_df = filtered_df[filtered_df['mode'] == 1]
    else:
        if target_valence >= 0.50: 
            genres_to_search = ['ambient', 'piano', 'new-age', 'jazz']
            if 'mode' in filtered_df.columns: filtered_df = filtered_df[filtered_df['mode'] == 1]
        else: 
            genres_to_search = ['ambient', 'classical']

    print(f"[FILTER] Szukam wsrod gatunkow: {genres_to_search}")

    pattern = '|'.join(genres_to_search)
    candidates = filtered_df[filtered_df['track_genre'].str.contains(pattern, case=False, na=False)].copy()
    if 'instrumentalness' in candidates.columns: candidates = candidates[candidates['instrumentalness'] > 0.5]

    tolerance = 0.1
    result = pd.DataFrame()
    while result.empty and tolerance <= 0.6:
        result = candidates[
            (candidates['valence'].between(target_valence - tolerance, target_valence + tolerance)) &
            (candidates['energy'].between(target_energy - tolerance, target_energy + tolerance))
        ]
        tolerance += 0.05
    
    final = None
    if result.empty:
        print("[FALLBACK] Nie znaleziono idealnego utworu, biore losowy z pasujacych gatunkow.")
        final = candidates.sample(1).iloc[0] if not candidates.empty else df.sample(1).iloc[0]
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

PRESET_SCENARIOS = {
    "SCENARIUSZ 1 (The Speckled Band)": 
    # EMOCJA: GNIEW (Dr Roylott)
    "“Don’t you dare to meddle with my affairs. I know that Miss Stoner has been here. I traced her! I am a dangerous man to fall foul of! See here.” He stepped swiftly forward, seized the poker, and bent it into a curve with his huge brown hands. “See that you keep yourself out of my grip,” he snarled, and hurling the twisted poker into the fireplace he strode out of the room.",
    
    "SCENARIUSZ 2 (The Speckled Band)": 
    # EMOCJA: STRACH (Helen Stoner)
    "“It is fear, Mr. Holmes. It is terror.” She raised her veil as she spoke, and we could see that she was indeed in a pitiable state of agitation, her face all drawn and grey, with restless, frightened eyes, like those of some hunted animal. “It was the whistle,” she whispered. “In the dead of the night, amidst the howling of the gale, I heard that low, clear whistle again. I sprang from my bed, my limbs shaking with terror. I stood there in the darkness, paralyzed, unable to move, waiting for the dread thing that had killed my sister.”",
    
    "SCENARIUSZ 3 (The Beryl Coronet)": 
    # EMOCJA: SMUTEK (Bankier Holder)
    "“I am a ruined man, Mr. Holmes—a ruined man!” he cried, wringing his hands in an agony of despair. “My honour is gone, my good name is lost... I kept the treasure in my own hands, and it is gone. God help me! God help me!” He put his hands to his face and rocked himself to and fro, tears streaming down his cheeks.",
    
    "SCENARIUSZ 4 (The Man with the Twisted Lip)": 
    # EMOCJA: CZULOSC (Spotkanie malzonkow)
    "The woman rushed forward and threw her arms round the man’s neck. “Neville!” she cried, “I knew that you were safe! I knew that you were safe!” She held him close with a devotion that spoke more than words. “I am safe, my love,” he murmured, stroking her hair. In that moment of reunion, the grim surroundings seemed to fade away.",
    
    "SCENARIUSZ 5 (A Study in Scarlet)": 
    # EMOCJA: RADOSC (Klasyczny moment triumfu)
    "“We have done it!” he cried, clapping his hands together. “The mystery is solved! The case is clear. It is a masterpiece!” His face radiated happiness and excitement, the heavy burden of the investigation lifting to reveal a spirit light and jubilant."
}

def main():
    # --- PASEK BOCZNY ---
    with st.sidebar:
        st.title("Panel Badawczy")
        
        if 'session_results' not in st.session_state:
            st.session_state.session_results = []
        if 'umux_results' not in st.session_state:
            st.session_state.umux_results = {}

        # BLOCK RANDOMIZATION
        if 'test_queue' not in st.session_state:
            queue = ['ALGO'] * 5 + ['RANDOM'] * 5
            random.shuffle(queue)
            st.session_state.test_queue = queue
            print(f"[SYSTEM] Utworzono kolejke testowa: {st.session_state.test_queue}")
            
        count = len(st.session_state.session_results)
        st.metric("Liczba testow", count)
        st.progress(min(count / 5, 1.0))
        
        st.markdown("---")
        st.markdown("### Instrukcja:")
        # Zmiana: Instrukcja pasuje teraz do nazw przycisków
        st.markdown("""
        1. Kliknij **Wylosuj Tekst i Muzyke (START)**.
        2. Przeczytaj tekst słuchając muzyki.
        3. Oceń dopasowanie.
        4. Wykonaj min. **5 prób**.
        5. Kliknij **Zakoncz i ocen system (UMUX-Lite)**.
        """)
        
        if count >= 5 and st.session_state.phase != 'UMUX' and st.session_state.phase != 'DOWNLOAD':
             if st.button("Zakoncz i ocen system (UMUX-Lite)", type="primary"):
                 print("[UI] Kliknieto: Zakoncz i ocen (UMUX)")
                 st.session_state.phase = 'UMUX'
                 st.rerun()

    # --- GLOWNE OKNO ---
    st.title("Sherlock Holmes: AI Music Experiment")
    
    # Inicjalizacja Stanow
    if 'phase' not in st.session_state: st.session_state.phase = 'INPUT'
    if 'rec_method' not in st.session_state: st.session_state.rec_method = None
    if 'track' not in st.session_state: st.session_state.track = None
    if 'user_text' not in st.session_state: st.session_state.user_text = ""
    if 'scenario_name' not in st.session_state: st.session_state.scenario_name = ""

    # FAZA 1: WEJSCIE
    if st.session_state.phase == 'INPUT':
        
        # --- ROZBUDOWANY OPIS I INSTRUKCJA ---
        st.markdown("""
        **Co musisz zrobić?**
        System wylosuje dla Ciebie krótki fragment powieści oraz utwór muzyczny.
        Twoim zadaniem jest przeczytać tekst podczas słuchania i ocenić, czy muzyka pasuje do klimatu.
        """)
        # --------------------------------------------------------------------

        st.markdown("---")
        st.subheader("Gotowy? Zaczynamy!")
        
        if st.button("Wylosuj Tekst i Muzyke (START)", type="primary", use_container_width=True):
            print("[UI] Kliknieto START - Rozpoczynanie nowego testu")
            
            scenario_name, scenario_text = random.choice(list(PRESET_SCENARIOS.items()))
            st.session_state.user_text = scenario_text
            st.session_state.scenario_name = scenario_name
            print(f"[UI] Wylosowano scenariusz: {scenario_name}")
            
            if st.session_state.test_queue:
                method = st.session_state.test_queue.pop(0)
            else:
                method = random.choice(['ALGO', 'RANDOM'])
                
            st.session_state.rec_method = method
            print(f"[SYSTEM] Metoda wybrana z kolejki: {method}")
            
            with st.spinner("Analiza tekstu i dobieranie utworu..."):
                if method == 'ALGO':
                    time.sleep(0.5)
                    st.session_state.track = process_text_smart(scenario_text, nlp_classifier, fuzzy_system, spotify_df)
                else:
                    time.sleep(0.5)
                    st.session_state.track = get_random_track(spotify_df)
            
            st.session_state.phase = 'RATING'
            st.rerun()

    # FAZA 2: OCENA
    elif st.session_state.phase == 'RATING':
        st.subheader("2. Odsluch i Czytanie")
        
        track = st.session_state.track
        st.markdown("### Krok 1: Wlacz muzyke")
        
        embed_url = f"https://open.spotify.com/embed/track/{track['track_id']}?utm_source=generator"
        st.components.v1.iframe(embed_url, height=152)

        # --- PRZYCISK PONOWNEGO LOSOWANIA ---
        if st.button("Utwor nie dziala? Wylosuj inny zestaw (Tekst + Muzyka)"):
            print("[UI] Kliknieto: Utwor nie dziala (RESET)")
            current_method = st.session_state.rec_method
            print(f"[SYSTEM] Zachowuje metode: {current_method}")

            with st.spinner("Losowanie nowego zestawu..."):
                scenario_name, scenario_text = random.choice(list(PRESET_SCENARIOS.items()))
                st.session_state.user_text = scenario_text
                st.session_state.scenario_name = scenario_name
                
                if current_method == 'ALGO':
                    time.sleep(0.5)
                    st.session_state.track = process_text_smart(scenario_text, nlp_classifier, fuzzy_system, spotify_df)
                else:
                    time.sleep(0.5)
                    st.session_state.track = get_random_track(spotify_df)
            st.rerun()
        # ------------------------------------
        
        st.markdown("### Krok 2: Przeczytaj tekst")
        st.caption(f"Scena: {st.session_state.scenario_name}")
        st.markdown(f"> *\"{st.session_state.user_text}\"*")
        
        st.markdown("---")
        st.markdown("### Krok 3: Ocen wrazenia")
        st.write("Czy ta muzyka pasuje do emocji w tekscie?")
        
        with st.form("rating"):
            rating = st.slider("Ocena dopasowania", 1, 10, 5)
            
            # Podpisy pod suwakiem widoczne zawsze
            c1, c2, c3 = st.columns([1, 2, 1])
            with c1:
                st.caption("1 = Zupełnie nie pasuje")
            with c3:
                st.caption("10 = Idealnie pasuje")
            
            if st.form_submit_button("Zatwierdz Ocene", type="primary"):
                print(f"[UI] Zatwierdzono ocene: {rating}/10 dla metody {st.session_state.rec_method}")
                res = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'scenario': st.session_state.scenario_name,
                    'user_text': st.session_state.user_text,
                    'method': st.session_state.rec_method, 
                    'rating': rating,
                    'track_name': track['track_name'],
                    'artist': track['artists']
                }
                st.session_state.session_results.append(res)
                st.session_state.phase = 'REVEAL'
                st.rerun()

    # FAZA 3: WYNIK
    elif st.session_state.phase == 'REVEAL':
        st.subheader("3. Zapisano!")
        
        st.write(f"Odsluchany utwor: **{st.session_state.track['track_name']}** - {st.session_state.track['artists']}")
        
        col1, col2 = st.columns(2)
        with col1:
             if st.button("Kolejny Test", type="primary", use_container_width=True):
                print("[UI] Kliknieto: Kolejny Test")
                st.session_state.phase = 'INPUT'
                st.session_state.user_text = ""
                st.rerun()
        with col2:
            if len(st.session_state.session_results) >= 5:
                st.success("Mozesz teraz przejsc do ankiety UMUX-Lite.")
            else:
                remaining = 5 - len(st.session_state.session_results)
                st.caption(f"Zrob jeszcze {remaining} testy, aby zakonczyc.")

    # FAZA 4: UMUX
    elif st.session_state.phase == 'UMUX':
        st.subheader("Ankieta UMUX-Lite")
        st.write("Ocen sposób dobierania muzyki w skali 1-7 (1 = Zdecydowanie nie zgadzam sie, 7 = Zdecydowanie zgadzam sie).")
        
        with st.form("umux_form"):
            u1 = st.slider("1. Ten sposób dobierania muzyki spełnia moje wymagania.", 1, 7, 4)
            u2 = st.slider("2. Ten dobór muzyki jest łatwy w odbiorze/zrozumiały.", 1, 7, 4)
            
            if st.form_submit_button("Zapisz i Pobierz Wyniki", type="primary"):
                print(f"[UI] Zapisano UMUX: U1={u1}, U2={u2}")
                st.session_state.umux_results = {
                    'umux_1_capabilities': u1,
                    'umux_2_ease': u2
                }
                st.session_state.phase = 'DOWNLOAD'
                st.rerun()

    # FAZA 5: DOWNLOAD
    elif st.session_state.phase == 'DOWNLOAD':
        st.subheader("Dziekuje za udzial!")
        st.success("Badanie zakonczone.")
        
        final_data = []
        umux = st.session_state.umux_results
        
        for row in st.session_state.session_results:
            full_row = {**row, **umux} 
            final_data.append(full_row)
            
        df_final = pd.DataFrame(final_data)
        
        csv_data = df_final.to_csv(index=False).encode('utf-8')
        timestamp = datetime.now().strftime("%d-%m_%H-%M")
        
        print("[UI] Generowanie pliku CSV do pobrania.")
        st.download_button(
            label="POBIERZ PELNE WYNIKI (CSV)",
            data=csv_data,
            file_name=f"badanie_full_{timestamp}.csv",
            mime='text/csv',
            type="primary",
            use_container_width=True
        )
        
        st.markdown("---")
        st.markdown("#### Opcja na telefon (Kopiuj-Wklej):")
        st.code(df_final.to_csv(index=False), language='csv')

if __name__ == "__main__":
    main()