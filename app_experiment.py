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

    # 3. Norm 
    target_valence = max(0.0, min(1.0, (val_fuzzy - 1) / 8.0))
    target_energy = max(0.0, min(1.0, (aro_fuzzy - 1) / 8.0))

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
    "SCENARIUSZ 1 (Dracula)": 
    # EMOCJA: GNIEW (Hrabia Dracula wpada w furię)
    "I was conscious of the presence of the Count, and of his being as if lapped in a storm of fury. I saw his strong hand grasp the slender neck of the fair woman and with giant’s power draw it back, the blue eyes transformed with fury, the white teeth champing with rage, and the fair cheeks blazing red with passion. But the Count! Never did I imagine such wrath and fury; his eyes were positively blazing, as if the flames of hell-fire blazed behind them. His face was deathly pale, and the lines of it were hard like drawn wires; the thick eyebrows that met over the nose now seemed like a heaving bar of white-hot metal. With a fierce sweep of his arm, he hurled the woman from him, and then motioned to the others, as though he were beating them back; it was the same imperious gesture that I had seen used to the wolves. In a voice which seemed to cut through the air and then ring round the room he screamed: ‘How dare you touch him, any of you? How dare you cast eyes on him when I had forbidden it? Back, I tell you all! This man belongs to me! Beware how you meddle with him, or you’ll have to deal with me! My wrath shall know no bounds if you defy me!’",   
    
    "SCENARIUSZ 2 (The Speckled Band)": 
    # EMOCJA: STRACH (Helen Stoner szuka pomocy)
    "It is fear, Mr. Holmes. It is terror.” She raised her veil as she spoke, and we could see that she was indeed in a pitiable state of agitation, her face all drawn and grey, with restless frightened eyes, like those of some hunted animal. Her hair was shot with premature grey, and her expression was weary and haggard. The lady gave a violent start and stared in bewilderment, crying out: “Sir, I can stand this strain no longer; I shall go mad if it continues. I have no one to turn to—none, save only one, who cares for me, and he can be of little aid. Oh, sir, do you not think that you could help me, and at least throw a little light through the dense darkness which surrounds me? The very horror of my situation lies in the fact that my fears are so vague, and my suspicions depend so entirely upon small points, which might seem trivial to another. I am a victim of a terror that never sleeps, a dread that follows me even into my dreams. My nerves are worked up to the highest pitch of tension, and I feel as though some invisible hand is closing round my throat in the silence of the night. You may advise me how to walk amid the dangers which encompass me.",
    
    "SCENARIUSZ 3 (The Beryl Coronet)": 
    # EMOCJA: SMUTEK (Złamany bankier)
    "He was a man of about fifty, tall, portly, and imposing, but his face was ash-coloured and his breath came in short gasps. I looked at him, and then at my companion, and knew at once that our visitor was a man whose spirit had been completely broken. 'I am a ruined man, Mr. Holmes—a ruined man!' he cried, wringing his hands in an agony of despair. 'My honour is gone, my good name is lost, and I have to face my family with the knowledge that I have brought them to shame. God help me! God help me!' He put his hands to his face and rocked himself to and fro, tears streaming down his cheeks, the picture of a man whom some sudden and terrible misfortune has overwhelmed. Every heavy sigh that escaped him seemed to mark the death of his former self, leaving only a hollow shell of grief, from which there could be no recovery. The world has grown dark and empty, and I feel as though I am sinking into a deep, cold abyss of sorrow where no light can reach me. My heart is heavy as lead, and I can only weep for what is lost and can never be regained.",
    
    "SCENARIUSZ 4 (A Little Princess)": 
    # EMOCJA: CZUŁOŚĆ (Sara i Becky)
    "Suddenly—and it was all through the loving mournfulness of Becky's streaming eyes—Sara held out her hand and gave a little sob. 'Oh, Becky,' she said. 'I told you we were just the same—only two little girls—just two little girls. You see how true it is. There's no difference now.' Becky ran to her and caught her hand, and hugged it to her breast, kneeling beside her and sobbing with love and pain. 'Yes, miss, you are,' she cried, and her words were all broken. 'Whats'ever 'appens to you—whats'ever—you'd be a princess all the same—an' nothin' couldn't make you nothin' different.' Sara felt a deep, quiet devotion to her little friend, and her heart was full of a strange, gentle tenderness as she stroked Becky's hair with a soft touch. They sat together in the silence of the attic, two small souls finding peace in each other’s company. It was a moment of such pure intimacy that the cold room seemed to grow warm with the light of a faithful, loving heart.",
    
    "SCENARIUSZ 5 (A Little Princess)": 
    # EMOCJA: RADOŚĆ / TRIUMF (Magia na poddaszu)
    "Her eyes opened in spite of herself, and then she actually smiled—for what she saw she had never seen in the attic before. She put her feet on the floor with a rapturous smile. 'I am dreaming it stays—real! I'm dreaming it FEELS real!' Her face was a shining, wonderful thing. 'It's true! It's true!' she cried. 'I've touched them all. They are as real as we are. The Magic has come and done it, Becky!' Sara stood in the warm, glowing midst of things, her heart leaping with a joy she had never known. It was a moment of unalloyed happiness and triumph, as if the world had suddenly turned into fairyland. 'I am NOT dreaming!' she cried aloud, laughing with sheer delight at the wonderful magic that had happened. Her eyes sparkled with a jubilant light, and she began to dance around the room, clapping her hands in a state of high spirits. It was a little triumph to make the dream complete, a brilliant discovery that cleared up the darkness of her life and filled her soul with an unexpected, life-giving energy."
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
    st.title("Music Experiment")
    
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
        (Opcja pobrania wyników na telefonie jest w lewej górnej strzałce)
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
