
# Alle notwendigen Importe
import streamlit as st
import openai
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras import preprocessing
import time

# Der Verweis auf unseren OpenAI Secret Key damit dieser nicht direkt im Quellcode steht
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Der Titel unserer Anwendung
st.title('Simpsons Charakter Klassifizierer')

# Unterüberschrift
st.markdown("Nutze dieses AI Tool um Bilder deiner Simpsons Charaktere klassifizieren zu lassen")

# Unsere Main Funktion welche alles ausführt
def main():
    # Unsere Bilder Hochlade Funktion mit Streamlit
    file_uploaded = st.file_uploader("Lade ein Bild eines Simpsons Charakters hoch", type=["png", "jpg", "jpeg"])
    # Der Button zum Klassifizieren
    class_btn = st.button("Klassifizieren")
    # If Funktion schaut wenn Bild hochgeladen wurde und gibt dieses dann aus
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.image(image.resize((200,200)), caption='Hochgeladenes Bild', use_column_width=True)
    # If Funktion die Schaut ob eine Korrekte Bild Eingabe getätigt wurde und wenn ja startet die predict Funktion
    if class_btn:
        if file_uploaded is None:
            st.write("Falsche Eingabe, bitte lade ein Bild mit dem Format jpg, png oder jpeg hoch")
        else:
            with st.spinner('Das Modell berechnet....'):
                # Die unten aufgestellte predict Funktion wird mit dem übergebenen Bild gestartet
                predictions = predict(image)

                time.sleep(1)
                st.success('Erfolgreich Klassifiziert')
                # Ausgabe der predict Funktion
                st.write(predictions)
    # Die Open AI Chatbot Funktionalität
    st.title("AI Chatbot")
    st.markdown("Du hast Fragen stell sie dem AI Chatbot")
    # Wählen des ChatGPT Modells und Key
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"
    #
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Die Nutzerrolle wird definiert
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # Der User Chat Input und eingegebene Frage wird gespeichert und ausgegeben
    prompt = st.chat_input("Wie kann ich behilflich sein ?")
    if prompt:

        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})

        # Die OpenAI Chatbot Funktion
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True
            ):
                # Ausgabe und Speicherung der Chsatbot Antwort
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response + " ")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})


# Die Funktion die die hochgeladenen Bilder Klassifiziert
def predict(image):
    # Unser abgespeichertes Simpsons CNN Modell
    classifier_model ="simpsons_classifier_epoch_83.h5"
    model = load_model(classifier_model)
    # Hochgeladene Bilder werden größentechnisch angepasst um sie als Input für das Modell nutzen zu können
    test_image = image.resize((224,224))
    test_image = preprocessing.image.img_to_array(test_image)
    # Farbwerte werden normiert
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    # Klassennamen werden festgelegt
    class_names = {0: 'abraham_grampa_simpsons',1: 'agnes_skinner',2: 'apu_nahasapeemapetilion',3:'barney_gumble',4:'bart_simpson',5:'carl_carlson',6:'charles_montgomery_burns',7:'chief_wiggum',8:'cletus_spuckler',9:'comic_book_guy',10:'disco_stu',11:'edna_krabappel',12:'fat_tony',13:'gil',14:'groundskeeper_willie',15:'homer_simpson',16:'kent_brockman',17:'krusty_the_clown',18:'lenny_leonard',19:'lionel_hutz',20:'lisa_simpson',21:'maggie_simpson',22:'marge_simpson',23:'martin_prince',24:'mayor_quimby',25:'milhouse_van_houten',26:'miss_hoover',27:'moe_szyslak',28:'ned_flanders',29:'nelson_muntz',30:'otto_mann',31:'patty_bouvier',32:'principal_skinner',33:'professor_john_frink',34:'rainer_wolfcastle',35:'ralph_wiggum',36:'selma_bouvier',37:'sideshow_bob',38:'sideshow_mel',39:'snake_jailbird',40:'troy_mcclure',41:'waylon_smithers'}
    #class_names = {'abraham_grampa_simpson': 0, 'agnes_skinner': 1, 'apu_nahasapeemapetilon': 2, 'barney_gumble': 3, 'bart_simpson': 4, 'carl_carlson': 5, 'charles_montgomery_burns': 6, 'chief_wiggum': 7, 'cletus_spuckler': 8, 'comic_book_guy': 9, 'disco_stu': 10, 'edna_krabappel': 11, 'fat_tony': 12, 'gil': 13, 'groundskeeper_willie': 14, 'homer_simpson': 15, 'kent_brockman': 16, 'krusty_the_clown': 17, 'lenny_leonard': 18, 'lionel_hutz': 19, 'lisa_simpson': 20, 'maggie_simpson': 21, 'marge_simpson': 22, 'martin_prince': 23, 'mayor_quimby': 24, 'milhouse_van_houten': 25, 'miss_hoover': 26, 'moe_szyslak': 27, 'ned_flanders': 28, 'nelson_muntz': 29, 'otto_mann': 30, 'patty_bouvier': 31, 'principal_skinner': 32, 'professor_john_frink': 33, 'rainier_wolfcastle': 34, 'ralph_wiggum': 35, 'selma_bouvier': 36, 'sideshow_bob': 37, 'sideshow_mel': 38, 'snake_jailbird': 39, 'troy_mcclure': 40, 'waylon_smithers': 41}
    # Die Prediktion wird ausgeführt
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    # Die Warscheinlichkeit für das Ergebnis wird errechnet
    scores = scores.numpy()

    # Das Ergebnis der predict Funktion wird in der variabel result gespeichert mit der dem Namen der predizierten Klasse und deren Warscheinlichkeit
    result = f"Es handelt sich um {class_names[np.argmax(scores)]} mit einer { (100 * np.max(scores)).round(2) } % Warscheinlichkeit."
    return result




if __name__ == "__main__":
    main()
