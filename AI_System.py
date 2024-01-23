import os
import pyttsx3
import speech_recognition as sr
from deep_translator import GoogleTranslator, single_detection
from dotenv import load_dotenv
from openai import OpenAI
import whisper
from typing import Optional
from collections import namedtuple


class BiD_AI_System:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.ttx = pyttsx3.init()

        load_dotenv()
        self.LANGDETECT_API_KEY = os.environ["LANGDETECT_API_KEY"]

        # ask client what language they speak
        # this informs translation choice
        self.lang = input("What language do you speak? ").capitalize()

        print(f"You speak {self.lang}")

        # this is the language spoken by the volunteer, denoted by "en" in further lines
        self.volunteer_language = "English"

        # a list of language codes supported by Google's Speech-to-Text API
        Lang_Spec = namedtuple("Lang_Spec", {"Country", "language_code"})

        self.language_dictionary = {
            "Arabic": Lang_Spec("UAE", "ar-AE"),
            "Bengali": Lang_Spec("Bangladesh", "bn-BD"),
            "Bulgarian": Lang_Spec("Bulgaria", "bg-BG"),
            "Chinese Mandarin": Lang_Spec("China", "zh-TW-TW"),
            "Chinese Cantonese": Lang_Spec("Hong Kong", "yue-Hant-HK"),
            "Croatian": Lang_Spec("Croatia", "hr_HR"),
            "Czech": Lang_Spec("Czech Republic", "cs-CZ"),
            "Danish": Lang_Spec("Denmark", "da-DK"),
            "English": Lang_Spec("United Kingdom", "en-GB"),
            "Farsi": Lang_Spec("Iran", "fa-IR"),
            "French": Lang_Spec("France", "fr-FR"),
            "Filipino": Lang_Spec("Philippines", "fil-PH"),
            "German": Lang_Spec("Germany", "de-DE"),
            "Greek": Lang_Spec("Greece", "el-GR"),
            "Finnish": Lang_Spec("Finland", "fi-FI"),
            "Hebrew": Lang_Spec("Israel", "iw-IL"),
            "Hindi": Lang_Spec("India", "hi-IN"),
            "Hungarian": Lang_Spec("Hungary", "hu-HU"),
            "Indonesian": Lang_Spec("Indonesia", "id-ID"),
            "Icelandic": Lang_Spec("Iceland", "is-IS"),
            "Igbo": Lang_Spec("Nigeria", "ig-IG"),
            "Italian": Lang_Spec("Italy", "it-IT"),
            "Japanese": Lang_Spec("Japan", "ja-JP"),
            "Korean": Lang_Spec("Korea", "ko-KR"),
            "Lithuanian": Lang_Spec("Lithuania", "lt-LT"),
            "Malaysian": Lang_Spec("Malaysia", "ms-MY"),
            "Dutch": Lang_Spec("Netherlands", "nl-NL"),
            "Norwegian": Lang_Spec("Norway", "nb-NO"),
            "Polish": Lang_Spec("Poland", "pl-PL"),
            "Portuguese": Lang_Spec("Portugal", "pt-PT"),
            "Romanian": Lang_Spec("Romania", "ro-RO"),
            "Russian": Lang_Spec("Russia", "ru-RU"),
            "Serbian": Lang_Spec("Serbia", "sr-RS"),
            "Slovak": Lang_Spec("Slovakia", "sk-SK"),
            "Slovenian": Lang_Spec("Slovenia", "sl-SI"),
            "Spanish": Lang_Spec("Spain", "es-ES"),
            "Swedish": Lang_Spec("Sweden", "sv-SE"),
            "Thai": Lang_Spec("Thailand", "th-TH"),
            "Turkish": Lang_Spec("Turkey", "tr-TR"),
            "Twi": Lang_Spec("Ghana", "ak-AK"),
            "Ukrainian": Lang_Spec("Ukraine", "uk-UA"),
            "Vietnamese": Lang_Spec("Viet Nam", "vi-VN"),
            "Yoruba": Lang_Spec("Nigeria", "yo-YO"),
            "Zulu": Lang_Spec("South Africa", "zu-ZA"),
        }

    # this is used to obtain the language code based on client's choice of language
    def get_spoken_language(self):
        """
        Obtain language code given spoken language
        """
        for language, country_code in self.language_dictionary.items():
            if language == self.lang:
                lan_code = self.language_dictionary[language].language_code.split("-")[
                    0
                ]

                return lan_code

    def rotation_translate(self, audio, language: str, target: str):
        """
        handles the translation of audio files
        between languages
        starts expecting English language audio,
        which is then translated to second language and vice versa
        """
        text = self.recognizer.recognize_google(
            audio_data=audio, language=language
        )  # convert audio file to text
        print(text)
        print("Detecting Language")

        input_language = single_detection(
            text, api_key=self.LANGDETECT_API_KEY
        )  # detect language in text
        print(f"Language is {input_language}")

        print(f"Translating to {self.lang} ...")
        translator = GoogleTranslator(
            source="auto", target=target
        )  # translate text to another language
        translation_out = translator.translate(text=text)
        print(translation_out)
        self.ttx.say(translation_out)  # speak text in new language
        self.ttx.runAndWait()

    def translate(self):
        """
        translates audio files between languages
        using a two-way system e.g. telephone
        """
        lang_code = self.get_spoken_language()
        print(f"Language code is {lang_code}")

        flag = True  # for tracking who is speaking. starts with volunteer speaking

        # this keeps running until deliberately stopped
        while True:
            with sr.Microphone() as source:
                print("Speak!")

                self.recognizer.adjust_for_ambient_noise(
                    source
                )  # block out background noise in audio
                audio = self.recognizer.listen(
                    source, timeout=10000
                )  # listen for audio signal

                try:
                    language, target = (
                        ("en", lang_code) if flag else (lang_code, "en")
                    )  # translates from english to 2nd language and vice vers
                    self.rotation_translate(audio, language, target)

                    flag = not flag

                except sr.WaitTimeoutError():
                    print("Idle ...")

                except sr.UnknownValueError():
                    print("Could not understand speech ...")

                except sr.RequestError as e:
                    print(f"Could not access translator: {e}")

    def summarize(self, text: str):
        """
        Summarizes a text using OpenAI ChatGPT API
        """
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a personal assistant",
                },
                {"role": "user", "content": f"Summarize this text: {text}"},
            ],
        )

        summaized_live_version = response.choices[0].message
        return summaized_live_version

    def transcribe(self, summary: bool = True, audio_file: Optional[str] = None):
        """
        transcribes audio conversations
        optionally provides a summary
        """

        live_speech = input("Should we transcribe now or later? Yes/No: ").capitalize()

        if live_speech == "Yes":  # if transcribing live speech
            with sr.Microphone() as source:
                print("Speak!")

                self.recognizer.adjust_for_ambient_noise(
                    source
                )  # block out background noise

                audio = self.recognizer.listen(
                    source, timeout=1000000
                )  # listen for sound signal

                live_text = self.recognizer.recognize_google(
                    audio, language="en"
                )  # transcribe speech to text

                # write transcript into a file
                with open("path_to_file", "w") as script:
                    script.write(live_text + "\n")

                if (
                    summary
                ):  # for generating summary of text, requires OpenAI API which isn't free.
                    summarized_live_version = self.summarize(live_text)
                    return summarized_live_version

                else:
                    return live_text

        else:  # if recorded audio file i.e NOT live
            model = whisper.load_model("tiny")  # load OpenAI Whisper model
            result = model.transcribe(
                audio_file,
                fp16=False,
            )  # transcribe audio file
            recorded_text = result["text"]

            # write transcript into a file
            with open("path_to_file", "w") as script:
                script.write(recorded_text + "\n")

            if (
                summary
            ):  # for generating summary of text, requires OpenAI API which isn't free.
                summarized_recorded_version = self.summarize(recorded_text)
                return summarized_recorded_version
            else:
                return recorded_text


if __name__ == "__main__": # select between speech transcription or language translation
    ais = BiD_AI_System()
    x = ais.transcribe(
        audio_file="path_to_'recorded'_audio_file",
        summary=False,
    )
    print(x)
    # # or
    # x = ais.translate()
