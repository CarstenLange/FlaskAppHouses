from flask import Flask, request, redirect, url_for, render_template
import math
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split


app = Flask(__name__)

@app.template_filter('comma_format')
def comma_format(value):
    try:
        return f"{float(value):,.0f}"
    except (ValueError, TypeError):
        return value

@app.route('/EnterHouseVars')
def choose_features():
    # Capture all current URL parameters
    existing_params = request.args.to_dict()
    
    # Check if the user has already submitted the dropdown selections
    if 'Sqft' in existing_params and 'Bedrooms' in existing_params:
        # Step 2: Push the full dictionary to your target function
        return redirect(url_for('FctSHAPResults', **existing_params))
    
    # Step 1: Otherwise, generate choices and show the user the form
    SqftChoices = list(range(1000, 4501, 200))
    BedroomChoices = list(range(1, 7))
    
    # Render using the standalone template file from the templates folder
    return render_template('FormSqftBedrooms.html', 
                           existing_params=existing_params,
                           SqftChoices=SqftChoices, 
                           BedroomChoices=BedroomChoices)
    

@app.route("/SHAPResults")
def FctSHAPResults():
    # request.args.to_dict() extracts incoming parameters as strings
    DictParams = request.args.to_dict()

   
    # Use .get() matching the exact keys in your URL string.
    # We cast to float() so they are ready for numerical analysis.
    DictParams["Price"] = float(DictParams.get("Price", 1000000))
    DictParams["Sqft"] = float(DictParams.get("Sqft", 2500))
    DictParams["Bedrooms"] = float(DictParams.get("Bedrooms", 3))    
    DictParams["BeachTime"] = float(DictParams.get("BeachTime", 47.8))
    DictParams["BeachTimeLN"] = math.log(DictParams["BeachTime"])
    DictParams["SchoolQuality"] = float(DictParams.get("SchoolQuality", 2678))
    DictParams["MedIncome"] = float(DictParams.get("MedIncome", 115461))
    DictParams["CrimesPerSqMile"] = float(DictParams.get("CrimesPerSqMile", 2.599))
    DictParams["DistParksMeters"] = float(DictParams.get("DistParksMeters", 9229))
    DictParams["DistParksMetersLN"] = math.log(DictParams["DistParksMeters"])
    DictParams["RestaurantsPerSqMile"] = float(DictParams.get("RestaurantsPerSqMile", 0.349))

    FctScaler=joblib.load("FctScaler.joblib")
    FctPCA=joblib.load("FctPCA.joblib")
    DataUserOrg=pd.DataFrame([DictParams])
    
    # 1. Define the exact order of features your model was trained on.
    # (Rearrange this list to match your original training set order!)
    FeatureOrderTrain = [
        "PCA1",
        "PCA2",
        "BeachTimeLN",
        "SchoolQuality",
        "MedIncome",
        "CrimesPerSqMile",
        "DistParksMetersLN",
        "RestaurantsPerSqMile",
    ]

    # 2. Build and explicitly align the DataFrame columns
    DataUserOrg = pd.DataFrame([DictParams])
    DataUserFinal = (
        DataUserOrg[["Sqft", "Bedrooms"]]
        .pipe(FctScaler.transform)
        .pipe(FctPCA.transform)
        .set_axis(["PCA1", "PCA2"], axis=1)
        .join([DataUserOrg])
        .drop(columns=["Sqft", "Bedrooms", "BeachTime", "DistParksMeters"])
        # Force the columns into the identical order used during model.fit()
        .reindex(columns=FeatureOrderTrain)
    )



    # 3. Load Explainer and Compute SHAP Values
    ExplainerOLSWithPCAs = joblib.load("ExplainerOLSWithPCAs.joblib")
    SHAPValuesUser=ExplainerOLSWithPCAs(DataUserFinal)
    SHAPValuesUserDict = dict(zip(FeatureOrderTrain, SHAPValuesUser.values[0]))

    # Prepare and Render to template

    
    DataAnalysis = pd.read_csv("DataAnalysis.csv").drop(columns=['SqftxBedrooms'])
  
    X_train, _, y_train, _ = train_test_split(
        DataAnalysis.drop(columns=['Price']),  # Features (X)
        DataAnalysis['Price'],                 # Target (y)
        test_size=0.2, 
        random_state=123,
        stratify=pd.qcut(DataAnalysis['Price'], q=5, labels=False)
    )

    SHAPPCA1=SHAPValuesUserDict.get("PCA1", 99999999)
    SHAPPCA2=SHAPValuesUserDict.get("PCA2", 99999999)
    SHAPPCA1PCA2 = SHAPPCA1 +SHAPPCA2


    return render_template("ResultsOLS.html", 
        PredPrice=float(ExplainerOLSWithPCAs.model.predict(DataUserFinal)[0]), 
        Sqft=DictParams["Sqft"],
        Bedrooms=DictParams["Bedrooms"],
        SchoolQuality=DictParams["SchoolQuality"],
        MedIncome=DictParams["MedIncome"],
        BeachTime=math.exp(DictParams["BeachTimeLN"]),
        RestaurantsPerSqMile=DictParams["RestaurantsPerSqMile"],
        CrimesPerSqMile=DictParams["CrimesPerSqMile"],
        SHAPBaseValue=float(SHAPValuesUser.base_values[0]),
        SHAPPCA1=SHAPPCA1,
        SHAPPCA2=SHAPPCA2,
        SHAPPCA1PCA2=SHAPPCA1PCA2,
        SHAPBeachTimeLN=SHAPValuesUserDict.get("BeachTimeLN", 99999999),
        SHAPSchoolQuality=SHAPValuesUserDict.get("SchoolQuality", 99999999),
        SHAPMedIncome=SHAPValuesUserDict.get("MedIncome", 99999999),
        SHAPRestaurantsPerSqMile=SHAPValuesUserDict.get("RestaurantsPerSqMile", 99999999),
        SHAPCrimesPerSqMile=SHAPValuesUserDict.get("CrimesPerSqMile", 99999999),
        AvgPrice=y_train.mean(),
        AvgSqft=X_train["Sqft"].mean(),                                              
        AvgBedrooms=X_train["Bedrooms"].mean(),
        GeoAvgBeachTime=math.exp(X_train["BeachTimeLN"].mean()),
        AvgSchoolQuality=X_train["SchoolQuality"].mean(),
        AvgMedIncome=X_train["MedIncome"].mean(),
        AvgRestaurantsPerSqMile=X_train["RestaurantsPerSqMile"].mean(),
        AvgCrimesPerSqMile=X_train["CrimesPerSqMile"].mean(),

                                              

)


if __name__ == "__main__":
    app.run(debug=True)