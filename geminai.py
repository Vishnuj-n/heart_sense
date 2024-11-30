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
    
    target:0 = No presence of heart disease
    target:1 = presence of heart disease
    sex:1 = Male
    sex:0 = Female
    age: <age of the human>
    machine_learning_model_accuracy:<accuracy of the model>
    """

    format = """
    #Patient Information:
    <basic information of patient in numbered  points >

    #Recommendations:
    doctor_consultation:<"if required `yes` else `no`">
    tests:<if required `yes` else `no`. if `yes` suggestions >
    physical_activity:<excerise/yoga/ recommended for heart patients if target ='1' else some good physical_activity >
    warning:<this is model prediction which can't replace the need for professional medical evaluation>

    #Reducing Impact (Patient Communication):
    < 5 numbered points on patient communication >
    """

    prompt = f"write ways to reduce impact and request patient if needed to doctor or tests needed to be taken consultation based on the instruction and input delimited in triple backticks```{input_data}```" + instruction + format
    
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text
