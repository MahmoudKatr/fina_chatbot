import speech_recognition as sr


def main():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print("Say something!")
        audio = r.listen(source)
        
        try:
            print("You have said: \n" + r.recognize_google(audio) )
            
        except Exception as e:
            print("error: \n" +str(e))
            
if  __name__ == "__main__":
    main()