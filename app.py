from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load your pre-trained model here
model = tf.keras.models.load_model('plant_identification_model.keras')

# Define your label mapping dictionary
label_mapping = {0: 'Aloevera', 1: 'Amla', 2: 'Amruthaballi', 3: 'Arali', 4: 'Astma_weed', 5: 'Badipala',
                 6: 'Balloon_Vine', 7: 'Bamboo', 8: 'Beans', 9: 'Betel', 10: 'Bhrami', 11: 'Bringaraja',
                 12: 'Caricature', 13: 'Castor', 14: 'Catharanthus', 15: 'Chakte', 16: 'Chilly',
                 17: 'Citron lime (herelikai)', 18: 'Coffee', 19: 'Common rue(naagdalli)', 20: 'Coriender',
                 21: 'Curry', 22: 'Doddpathre', 23: 'Drumstick', 24: 'Ekka', 25: 'Eucalyptus', 26: 'Ganigale',
                 27: 'Ganike', 28: 'Gasagase', 29: 'Ginger', 30: 'Globe Amarnath', 31: 'Guava', 32: 'Henna',
                 33: 'Hibiscus', 34: 'Honge', 35: 'Insulin', 36: 'Jackfruit', 37: 'Jasmine', 38: 'Kambajala',
                 39: 'Kasambruga', 40: 'Kohlrabi', 41: 'Lantana', 42: 'Lemon', 43: 'Lemongrass', 44: 'Malabar_Nut',
                 45: 'Malabar_Spinach', 46: 'Mango', 47: 'Marigold', 48: 'Mint', 49: 'Neem', 50: 'Nelavembu',
                 51: 'Nerale', 52: 'Nooni', 53: 'Onion', 54: 'Padri', 55: 'Palak(Spinach)', 56: 'Papaya', 57: 'Parijatha',
                 58: 'Pea', 59: 'Pepper', 60: 'Pomoegranate', 61: 'Pumpkin', 62: 'Raddish', 63: 'Rose', 64: 'Sampige',
                 65: 'Sapota', 66: 'Seethaashoka', 67: 'Seethapala', 68: 'Spinach1', 69: 'Tamarind', 70: 'Taro', 71: 'Tecoma',
                 72: 'Thumbe', 73: 'Tomato', 74: 'Tulsi', 75: 'Turmeric', 76: 'ashoka', 77: 'camphor', 78: 'kamakasturi', 79: 'kepala'}

# Define your plant uses dictionary
plant_uses = {
    'Aloevera': 'Used for skin treatment, digestive health, and wound healing.',
    'Amla': 'Rich in Vitamin C, used for boosting immunity, and improving hair and skin health.',
    'Amruthaballi': 'Used to treat fever, diabetes, and improve the immune system.',
    'Arali': 'Known for its use in traditional medicine for respiratory issues and skin ailments.',
    'Astma_weed': 'Used for treating asthma and respiratory problems.',
    'Badipala': 'Utilized in traditional remedies for digestion and inflammation.',
    'Balloon_Vine': 'Used to treat arthritis, joint pain, and skin diseases.',
    'Bamboo': 'Used in traditional medicine for respiratory disorders and to treat infections.',
    'Beans': 'Rich in proteins and vitamins, good for overall health.',
    'Betel': 'Used in traditional medicine to treat headache, cold, and improve digestion.',
    'Bhrami': 'Known for improving memory and reducing anxiety.',
    'Bringaraja': 'Used to promote hair growth and improve liver function.',
    'Caricature': 'Used for anti-inflammatory and pain relief purposes.',
    'Castor': 'Used as a laxative and to induce labor.',
    'Catharanthus': 'Used in the treatment of cancer, diabetes, and high blood pressure.',
    'Chakte': 'Known for treating skin diseases and wounds.',
    'Chilly': 'Rich in vitamins, used to boost metabolism and reduce pain.',
    'Citron lime (herelikai)': 'Used to treat digestive disorders and improve skin health.',
    'Coffee': 'Used as a stimulant to increase alertness and reduce fatigue.',
    'Common rue(naagdalli)': 'Used to treat respiratory and digestive issues.',
    'Coriender': 'Rich in antioxidants, used to improve digestion and lower blood sugar levels.',
    'Curry': 'Used for its anti-inflammatory properties and to improve digestion.',
    'Doddpathre': 'Used to treat respiratory infections and digestive problems.',
    'Drumstick': 'Rich in vitamins, used to improve bone health and treat anemia.',
    'Ekka': 'Used in traditional medicine for its anti-inflammatory properties.',
    'Eucalyptus': 'Used to treat respiratory problems and relieve pain.',
    'Ganigale': 'Used to treat digestive disorders and improve skin health.',
    'Ganike': 'Known for its use in treating respiratory and skin diseases.',
    'Gasagase': 'Used to improve sleep and treat respiratory problems.',
    'Ginger': 'Used to treat nausea, digestive issues, and inflammation.',
    'Globe Amarnath': 'Used for its anti-inflammatory and analgesic properties.',
    'Guava': 'Rich in vitamins, used to improve digestion and boost immunity.',
    'Henna': 'Used for hair conditioning and to treat skin conditions.',
    'Hibiscus': 'Used to lower blood pressure and improve liver health.',
    'Honge': 'Used to treat skin diseases and wounds.',
    'Insulin': 'Used to manage diabetes and lower blood sugar levels.',
    'Jackfruit': 'Rich in vitamins, used to boost immunity and improve digestion.',
    'Jasmine': 'Used for its calming effects and to improve skin health.',
    'Kambajala': 'Known for treating digestive issues and respiratory problems.',
    'Kasambruga': 'Used to treat skin diseases and improve digestion.',
    'Kohlrabi': 'Rich in vitamins, used to boost immunity and improve digestion.',
    'Lantana': 'Used to treat respiratory and skin diseases.',
    'Lemon': 'Rich in Vitamin C, used to improve immunity and skin health.',
    'Lemongrass': 'Used to relieve pain and improve digestion.',
    'Malabar_Nut': 'Used to treat respiratory diseases and improve digestion.',
    'Malabar_Spinach': 'Rich in vitamins, used to improve digestion and boost immunity.',
    'Mango': 'Rich in vitamins, used to improve digestion and boost immunity.',
    'Marigold': 'Used for its anti-inflammatory and wound-healing properties.',
    'Mint': 'Used to improve digestion and relieve respiratory problems.',
    'Neem': 'Used to treat skin diseases and improve oral health.',
    'Nelavembu': 'Used to treat fever and improve immunity.',
    'Nerale': 'Known for treating digestive issues and improving skin health.',
    'Nooni': 'Used to improve digestion and boost immunity.',
    'Onion': 'Rich in antioxidants, used to improve heart health and boost immunity.',
    'Padri': 'Used to treat respiratory problems and improve digestion.',
    'Palak(Spinach)': 'Rich in iron, used to treat anemia and improve digestion.',
    'Papaya': 'Rich in vitamins, used to improve digestion and boost immunity.',
    'Parijatha': 'Used to treat respiratory and skin diseases.',
    'Pea': 'Rich in proteins, used to improve digestion and boost immunity.',
    'Pepper': 'Used to improve digestion and treat respiratory problems.',
    'Pomoegranate': 'Rich in antioxidants, used to improve heart health and boost immunity.',
    'Pumpkin': 'Rich in vitamins, used to improve digestion and boost immunity.',
    'Raddish': 'Used to improve digestion and treat respiratory problems.',
    'Rose': 'Used for its anti-inflammatory and wound-healing properties.',
    'Sampige': 'Known for treating respiratory and skin diseases.',
    'Sapota': 'Rich in vitamins, used to improve digestion and boost immunity.',
    'Seethaashoka': 'Used to treat menstrual disorders and improve skin health.',
    'Seethapala': 'Rich in vitamins, used to improve digestion and boost immunity.',
    'Spinach1': 'Rich in iron, used to treat anemia and improve digestion.',
    'Tamarind': 'Used to improve digestion and treat respiratory problems.',
    'Taro': 'Rich in vitamins, used to improve digestion and boost immunity.',
    'Tecoma': 'Used to treat respiratory and skin diseases.',
    'Thumbe': 'Used to improve digestion and treat respiratory problems.',
    'Tomato': 'Rich in vitamins, used to improve heart health and boost immunity.',
    'Tulsi': 'Used to treat respiratory problems and improve digestion.',
    'Turmeric': 'Used for its anti-inflammatory properties and to improve skin health.',
    'ashoka': 'Used to treat menstrual disorders and improve skin health.',
    'camphor': 'Used to relieve pain and improve respiratory health.',
    'kamakasturi': 'Used to improve digestion and boost immunity.',
    'kepala': 'Known for treating digestive issues and respiratory problems.'
}

# Define your plant scientific names dictionary
plant_scientific_names = {
    'Aloevera': 'Aloe barbadensis miller',
    'Amla': 'Phyllanthus emblica',
    'Amruthaballi': 'Tinospora cordifolia',
    'Arali': 'Nerium oleander',
    'Astma_weed': 'Euphorbia hirta',
    'Badipala': 'Diospyros montana',
    'Balloon_Vine': 'Cardiospermum halicacabum',
    'Bamboo': 'Bambusoideae',
    'Beans': 'Phaseolus vulgaris',
    'Betel': 'Piper betle',
    'Bhrami': 'Bacopa monnieri',
    'Bringaraja': 'Eclipta prostrata',
    'Caricature': 'Graptophyllum pictum',
    'Castor': 'Ricinus communis',
    'Catharanthus': 'Catharanthus roseus',
    'Chakte': 'Cinnamomum tamala',
    'Chilly': 'Capsicum annuum',
    'Citron lime (herelikai)': 'Citrus medica',
    'Coffee': 'Coffea',
    'Common rue(naagdalli)': 'Ruta graveolens',
    'Coriender': 'Coriandrum sativum',
    'Curry': 'Murraya koenigii',
    'Doddpathre': 'Plectranthus amboinicus',
    'Drumstick': 'Moringa oleifera',
    'Ekka': 'Calotropis gigantea',
    'Eucalyptus': 'Eucalyptus globulus',
    'Ganigale': 'Cyclea peltata',
    'Ganike': 'Thottea siliquosa',
    'Gasagase': 'Papaver somniferum',
    'Ginger': 'Zingiber officinale',
    'Globe Amarnath': 'Gomphrena globosa',
    'Guava': 'Psidium guajava',
    'Henna': 'Lawsonia inermis',
    'Hibiscus': 'Hibiscus rosa-sinensis',
    'Honge': 'Pongamia pinnata',
    'Insulin': 'Costus pictus',
    'Jackfruit': 'Artocarpus heterophyllus',
    'Jasmine': 'Jasminum',
    'Kambajala': 'Careya arborea',
    'Kasambruga': 'Scoparia dulcis',
    'Kohlrabi': 'Brassica oleracea',
    'Lantana': 'Lantana camara',
    'Lemon': 'Citrus limon',
    'Lemongrass': 'Cymbopogon citratus',
    'Malabar_Nut': 'Justicia adhatoda',
    'Malabar_Spinach': 'Basella alba',
    'Mango': 'Mangifera indica',
    'Marigold': 'Tagetes',
    'Mint': 'Mentha',
    'Neem': 'Azadirachta indica',
    'Nelavembu': 'Andrographis paniculata',
    'Nerale': 'Syzygium cumini',
    'Nooni': 'Morinda citrifolia',
    'Onion': 'Allium cepa',
    'Padri': 'Cleistanthus collinus',
    'Palak(Spinach)': 'Spinacia oleracea',
    'Papaya': 'Carica papaya',
    'Parijatha': 'Nyctanthes arbor-tristis',
    'Pea': 'Pisum sativum',
    'Pepper': 'Piper nigrum',
    'Pomoegranate': 'Punica granatum',
    'Pumpkin': 'Cucurbita pepo',
    'Raddish': 'Raphanus sativus',
    'Rose': 'Rosa',
    'Sampige': 'Michelia champaca',
    'Sapota': 'Manilkara zapota',
    'Seethaashoka': 'Saraca asoca',
    'Seethapala': 'Annona squamosa',
    'Spinach1': 'Spinacia oleracea',
    'Tamarind': 'Tamarindus indica',
    'Taro': 'Colocasia esculenta',
    'Tecoma': 'Tecoma stans',
    'Thumbe': 'Leucas aspera',
    'Tomato': 'Solanum lycopersicum',
    'Tulsi': 'Ocimum tenuiflorum',
    'Turmeric': 'Curcuma longa',
    'ashoka': 'Saraca asoca',
    'camphor': 'Cinnamomum camphora',
    'kamakasturi': 'Ocimum basilicum',
    'kepala': 'Indigofera tinctoria'
}

def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array, image

def process_predictions(predictions):
    predicted_label_index = np.argmax(predictions)
    predicted_label = label_mapping.get(predicted_label_index, 'Unknown')
    confidence = predictions[0][predicted_label_index]
    plant_use = plant_uses.get(predicted_label, 'No use information available.')
    plant_scientific_name = plant_scientific_names.get(predicted_label, 'Unknown')
    return predicted_label, confidence, predictions[0], plant_use, plant_scientific_name

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    uploaded_image = request.files['image']
    if uploaded_image.filename != '':
        image = Image.open(uploaded_image)
        preprocessed_image_array, preprocessed_image_pil = preprocess_image(image)

        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        preprocessed_buffered = io.BytesIO()
        preprocessed_image_pil.save(preprocessed_buffered, format="JPEG")
        preprocessed_img_str = base64.b64encode(preprocessed_buffered.getvalue()).decode()

        predictions = model.predict(preprocessed_image_array)
        predicted_label, confidence, all_predictions, plant_use, plant_scientific_name = process_predictions(predictions)

        plt.figure(figsize=(10, 5))
        plt.bar(range(len(label_mapping)), all_predictions)
        plt.xticks(range(len(label_mapping)), list(label_mapping.values()), rotation=90)
        plt.xlabel('Plant Species')
        plt.ylabel('Prediction Confidence')
        plt.title(f'Prediction: {predicted_label} (Confidence: {confidence:.2f})')
        plt.tight_layout()
        plot_buffer = io.BytesIO()
        plt.savefig(plot_buffer, format='png')
        plot_buffer.seek(0)
        plot_img_str = base64.b64encode(plot_buffer.getvalue()).decode()

        return render_template('result.html',
                               result=f'Predicted Label: {predicted_label}, Confidence: {confidence:.2f}',
                               uploaded_image=img_str,
                               preprocessed_image=preprocessed_img_str,
                               plot_image=plot_img_str,
                               plant_use=plant_use,
                               plant_scientific_name=plant_scientific_name)
    else:
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
