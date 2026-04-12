# 🍔 Food Calorie Predictor (Machine Learning Project)

## 📌 Project Overview
This project predicts the calorie value of food items using Machine Learning models.  
The user enters a food name, and the system estimates its calories.

---

## 🎯 Objective
- Predict calories based on food name
- Compare Decision Tree and Neural Network models
- Build a simple interactive UI

---

## 📊 Dataset
- Source: Kaggle (Indian Food Nutrition Dataset)
- Type: Real-world dataset
- Features used:
  - Dish Name
  - Calories
  - Protein
  - Carbohydrates
  - Fat

---

## ⚙️ Technologies Used
- Python
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Streamlit

---

## 🔄 Workflow
1. Load dataset  
2. Data preprocessing (cleaning + column selection)  
3. Convert text to numeric using TF-IDF  
4. Train models:
   - Decision Tree  
   - Neural Network  
5. Predict calories from user input  
6. Evaluate using MAE (Mean Absolute Error)

---

## 🤖 Models Used

### 1. Decision Tree Regressor
- Works well on small datasets  
- Easy to interpret  

### 2. Neural Network
- Multi-layer model  
- Learns complex patterns  

---

## 📈 Evaluation Metric
- MAE (Mean Absolute Error)

---

## 💻 Output
- User enters food name  
- System predicts calories using both models  
- Displays prediction and error comparison  

---

## ⚠️ Limitations
- Uses only food name as input  
- Does not consider preparation method  
- Same food can have different calorie values  

---

## 🚀 Future Scope
- Use image-based food recognition  
- Include nutritional features as input  
- Improve accuracy with larger datasets  

---

## 👨‍💻 Author
Mukul Goyal
