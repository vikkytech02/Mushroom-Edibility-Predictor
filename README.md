# 🍄 Mushroom Edibility Predictor

A machine learning web application that predicts whether a mushroom is **edible** or **poisonous** based on its features using a Random Forest model. The app is built with **Streamlit** and trained using **scikit-learn**.

---

## 🧠 Model Overview

- **Algorithm:** Random Forest Classifier  
- **Dataset:** UCI Mushroom Classification Dataset  
- **Interface:** Streamlit Web App  
- **Features Used:** Cap shape, surface, color, bruises, odor, gill details, stalk characteristics, ring, spore print, habitat, etc.

---

## 🧪 Test Cases

### ✅ Edible Mushroom Example
- Cap Shape: Bell  
- Cap Surface: Smooth  
- Cap Color: White  
- Bruises: False  
- Odor: Almond  
- Gill Size: Broad  
- Gill Color: Pink  
- Habitat: Woods  

### ❌ Poisonous Mushroom Example
- Cap Shape: Convex  
- Cap Surface: Scaly  
- Cap Color: Yellow  
- Bruises: True  
- Odor: Foul  
- Gill Size: Narrow  
- Gill Color: Green  
- Habitat: Urban  

---

## 📁 Project Structure

```
mushroom-edibility-predictor/
├── mushrooms.csv                 # Dataset
├── train_model.py      # Training script
├── app.py                       # Streamlit app
├── requirements.txt             # Dependencies
└── README.md                    # Project documentation
```

---

## ⚙️ How to Run

1. **Clone the repository**

```bash
git clone https://github.com/vikkytech02/Mushroom-Edibility-Predictor.git
cd mushroom-edibility-predictor
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Train the model (if not already trained)**

```bash
python train_model.py
```

4. **Launch the app**

```bash
streamlit run app.py
```

---

## 🧾 Notes

- Dropdowns in the UI show full feature names (e.g., "Bell" instead of "b").
- Code-to-name mappings are maintained and saved with the model.
- Easy-to-use 3-column layout for user-friendly input.

---

## 📜 License

MIT License

---

## 🙏 Acknowledgements

- UCI Machine Learning Repository – Mushroom Dataset  
- Streamlit for the simple and elegant app UI  
- scikit-learn for ML model training and evaluation
