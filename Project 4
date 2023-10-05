import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
merged_df = pd.DataFrame()
csv_files = ["C:\Prakash S\Project\DDW_B18_0800_NIC_FINAL_STATE_RAJASTHAN-2011.csv", "C:\Prakash S\Project\DDW_B18_1200_NIC_FINAL_STATE_ARUNACHAL_PRADESH-2011.csv", "C:\Prakash S\Project\DDW_B18_1400_NIC_FINAL_STATE_MANIPUR-2011.csv", "C:\Prakash S\Project\DDW_B18_1500_NIC_FINAL_STATE_MIZORAM-2011.csv", "C:\Prakash S\Project\DDW_B18_1900_NIC_FINAL_STATE_WEST_BENGAL-2011.csv", "C:\Prakash S\Project\DDW_B18sc_0700_NIC_FINAL_STATE_NCT_OF_DELHI-2011.csv", "C:\Prakash S\Project\DDW_B18sc_1600_NIC_FINAL_STATE_TRIPURA-2011.csv", "C:\Prakash S\Project\DDW_B18sc_2000_NIC_FINAL_STATE_JHARKHAND-2011.csv", "C:\Prakash S\Project\DDW_B18sc_2400_NIC_FINAL_STATE_GUJARAT-2011.csv", "C:\Prakash S\Project\DDW_B18sc_2700_NIC_FINAL_STATE_MAHARASHTRA-2011.csv", "C:\Prakash S\Project\DDW_B18sc_2900_NIC_FINAL_STATE_KARNATAKA-2011.csv", "C:\Prakash S\Project\DDW_B18sc_3000_NIC_FINAL_STATE_GOA-2011.csv", "C:\Prakash S\Project\DDW_B18sc_3200_NIC_FINAL_STATE_KERALA-2011.csv", "C:\Prakash S\Project\DDW_B18sc_3300_NIC_FINAL_STATE_TAMIL_NADU-2011.csv", "C:\Prakash S\Project\DDW_B18sc_3400_NIC_FINAL_STATE_PUDUCHERRY-2011.csv", "C:\Prakash S\Project\DDW_B18st_0200_NIC_FINAL_STATE_HIMACHAL_PRADESH-2011.csv", "C:\Prakash S\Project\DDW_B18st_0500_NIC_FINAL_STATE_UTTARAKHAND-2011.csv", "C:\Prakash S\Project\DDW_B18st_0900_NIC_FINAL_STATE_UTTAR_PRADESH-2011.csv", "C:\Prakash S\Project\DDW_B18st_1000_NIC_FINAL_STATE_BIHAR-2011.csv", "C:\Prakash S\Project\DDW_B18st_1100_NIC_FINAL_STATE_SIKKIM-2011.csv", "C:\Prakash S\Project\DDW_B18st_1300_NIC_FINAL_STATE_NAGALAND-2011.csv", "C:\Prakash S\Project\DDW_B18st_1800_NIC_FINAL_STATE_ASSAM-2011.csv", "C:\Prakash S\Project\DDW_B18st_2100_NIC_FINAL_STATE_ODISHA-2011.csv"]
for file in csv_files:
  df = pd.read_csv (file)
  merged_df = merged_df.append(df, ignore_index=True)
data = pd.DataFrame({
    'Category' == ['Retail', 'Poultry', 'Agriculture', 'Manufacturing'],
    'Description' == ["This is a retail store selling electronics.",
                    "This is a poultry farm specializing in fishing.",
                    "Our company focuses on agricultural products."
                    "Manufacturing plant for automotive components.",
                    ],
})
X_train, X_test, y_train, y_test = train_test_split(data['Description'], data['Category'], test_size=0.2, random_state=42)
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)
y_pred = clf.predict(X_test_tfidf)
print("Classification Report:")
print(classification_report(y_test, y_pred))
new_description = "We manufacture high-quality furniture."
new_description_tfidf = tfidf_vectorizer.transform([new_description])
predicted_category = clf.predict(new_description_tfidf)[0]
print(f"Predicted Category for the new business description: {predicted_category}")
