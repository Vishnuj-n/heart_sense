import os
import google.generativeai as genai





def configure_genai(api_key):
    """Configure Gemini AI with provided API key"""
    if not api_key:
        raise ValueError("API key cannot be empty")
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        raise Exception(f"Failed to configure Gemini AI: {str(e)}")
#genai.configure(api_key=GEMINI_API_KEY)
def get_health_recommendations(prediction, sex, age, accuracy):
    input_data = f"""
    target:{prediction}
    sex:{sex}
    age:{age}
    machine_learning_model_accuracy:{accuracy}
    """

    instruction = """
    you will be provided information with triple backticks.Write ways to reduce impact and request patient if needed to doctor or tests needed to be taken consultation and patient information with recommendation .
    target:0 = No presence of heart disease
    target:1 = presence of heart disease
    sex:1 = Male
    sex:0 = Female
    age: <age of the human>
    machine_learning_model_accuracy:<accuracy of the model>
    
    """

    output_format = """
    #Patient Information:
    <basic information of patient in numbered  points in natural language >

    #Recommendations:
    <adoctor_consultaio,tests in numbere points>
    doctor_consultation:<"if required `yes` else `no`">
    tests:<if required `yes` else `no`. if `yes` suggestions >
    physical_activity:<excerise/yoga/ recommended for heart patients if target ='1' else some good physical_activity >
    

    #Reducing Impact (Patient Communication):
    < 5 numbered points on patient communication >
    """

    prompt = instruction + output_format+f" '''{input_data}'''" 
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text
