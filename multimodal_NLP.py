import tkinter as tk
from tkinter import messagebox, simpledialog
from sentence_transformers import SentenceTransformer, util
import threading
import os
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import pyttsx3
import speech_recognition as sr
from sentiment import SentimentAnalyser
from fuzzywuzzy import fuzz
from fpdf import FPDF
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import language_tool_python
import wave
import contextlib

os.makedirs("responses", exist_ok=True)

class InterviewApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("700x700")
        self.root.title("AI Job Interview Assistant")
        self.root.config(bg="#f0f4f8")
        
        self.semantic_model = None
        self.sentiment_model = None
        self.grammar_tool = None
        self.recognizer = sr.Recognizer()
        self.tts = pyttsx3.init()
        self.models_loaded = False
        self.keyword_embeddings = {}

        self.show_welcome_screen()

    def show_welcome_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        welcome_label = tk.Label(
            self.root,
            text="🎙 Welcome to the AI Job Interview Assistant 🎯\n\nYou will be asked 10 questions.\nAnswer by voice or type. NLP and grammar feedback will follow.",
            font=("Arial", 14),
            wraplength=600,
            bg="#f0f4f8"
        )
        welcome_label.pack(pady=80)

        self.proceed_btn = tk.Button(
            self.root, 
            text="➡ Proceed to Interview", 
            command=self.start_loading, 
            bg="#4CAF50", 
            fg="white", 
            font=("Arial", 14, "bold"), 
            width=25, 
            height=2
        )
        self.proceed_btn.pack()

    def start_loading(self):
        # Disable button and show loading status
        self.proceed_btn.config(text="⏳ Loading AI Models...", state="disabled", bg="#9e9e9e")
        threading.Thread(target=self.load_models, daemon=True).start()

    def load_models(self):
        try:
            # Initialize questions and keywords for pre-encoding
            self.questions = [
                "Tell me about yourself.",
                "Why do you want this job?",
                "What are your strengths and weaknesses?",
                "Describe a challenge you've overcome.",
                "Where do you see yourself in five years?",
                "Why should we hire you?",
                "Tell me about a time you worked in a team.",
                "What is your greatest achievement?",
                "How do you handle stress or pressure?",
                "Do you have any questions for us?"
            ]

            self.keywords_dict = {
                "Tell me about yourself.": ["background", "experience", "education", "skills"],
                "Why do you want this job?": ["company", "values", "mission", "position"],
                "What are your strengths and weaknesses?": ["strength", "weakness", "improve", "growth"],
                "Describe a challenge you've overcome.": ["problem", "solution", "difficult", "overcome"],
                "Where do you see yourself in five years?": ["future", "goal", "career", "progression"],
                "Why should we hire you?": ["fit", "qualifications", "value", "unique"],
                "Tell me about a time you worked in a team.": ["collaboration", "teamwork", "role", "project"],
                "What is your greatest achievement?": ["success", "goal", "achievement", "impact"],
                "How do you handle stress or pressure?": ["stress", "cope", "pressure", "calm"],
                "Do you have any questions for us?": ["culture", "growth", "expectations", "team"]
            }

            # Load models
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.sentiment_model = SentimentAnalyser()
            
            # Pre-encode keywords for performance
            for q, kws in self.keywords_dict.items():
                self.keyword_embeddings[q] = [
                    self.semantic_model.encode(kw, convert_to_tensor=True) for kw in kws
                ]

            try:
                self.grammar_tool = language_tool_python.LanguageTool("en-US")
            except Exception:
                self.grammar_tool = None
            
            self.models_loaded = True
            self.root.after(0, self.setup_main_ui)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load models: {e}"))
            self.root.after(0, lambda: self.proceed_btn.config(text="➡ Retry Proceed", state="normal", bg="#4CAF50"))

    def setup_main_ui(self):
        self.current_q = 0
        self.responses = []

        for widget in self.root.winfo_children():
            widget.destroy()

        self.question_label = tk.Label(self.root, text=self.format_question(), font=('Arial', 16, 'bold'), wraplength=650, bg="#f0f4f8")
        self.question_label.pack(pady=20)

        self.start_btn = tk.Button(self.root, text="🎤 Answer with Voice", command=self.record_and_analyze, bg="#4CAF50", fg="white", font=('Arial', 12, 'bold'), width=25, height=2)
        self.start_btn.pack(pady=10)

        self.manual_btn = tk.Button(self.root, text="📝 Type Answer Instead", command=self.manual_input, bg="#2196F3", fg="white", font=('Arial', 12, 'bold'), width=25, height=2)
        self.manual_btn.pack(pady=10)

        self.next_btn = tk.Button(self.root, text="➡ Next Question", command=self.next_question, bg="#FFC107", fg="black", font=('Arial', 12, 'bold'), width=25, height=2)
        self.next_btn.pack(pady=10)

        self.report_btn = tk.Button(self.root, text="📄 Generate Report", command=self.generate_pdf, bg="#9C27B0", fg="white", font=('Arial', 12, 'bold'), width=25, height=2)
        self.report_btn.pack(pady=10)

        self.graph_btn = tk.Button(self.root, text="📊 Show NLP Graph", command=self.show_graph, bg="#FF5722", fg="white", font=('Arial', 12, 'bold'), width=25, height=2)
        self.graph_btn.pack(pady=10)

        self.transcript_label = tk.Label(self.root, text="Transcript:", font=('Arial', 12), wraplength=650, bg="#f0f4f8")
        self.transcript_label.pack(pady=10)

        self.sentiment_label = tk.Label(self.root, text="Sentiment:", font=('Arial', 12), bg="#f0f4f8")
        self.sentiment_label.pack()

        self.score_label = tk.Label(self.root, text="Confidence Score:", font=('Arial', 12), bg="#f0f4f8")
        self.score_label.pack()

    def format_question(self):
        return f"{self.current_q + 1}. {self.questions[self.current_q]}"

    def speak_question(self):
        self.tts.say(self.questions[self.current_q])
        self.tts.runAndWait()

    def record_and_analyze(self):
        threading.Thread(target=self._record_and_analyze).start()

    def _record_and_analyze(self):
        self.speak_question()
        self.transcript_label.config(text="🎙 Recording... Speak clearly...")

        fs = 16000
        duration = 10
        filename = f"responses/q{self.current_q+1}.wav"

        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()

        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.9
        audio_pcm = np.int16(audio * 32767)
        write(filename, fs, audio_pcm)

        self.transcript_label.config(text="🔁 Transcribing...")
        transcript = self.transcribe_google_file(filename)

        if transcript in ["[Unrecognized Speech]", "[Google API Error]"]:
            self.transcript_label.config(text=f"⚠ {transcript}")
            return

        feedback = self.analyze_speech_rate_and_pauses(filename, transcript)
        self.root.after(0, lambda: self.edit_transcript_and_analyze(transcript, feedback))

    def edit_transcript_and_analyze(self, transcript, feedback):
        messagebox.showinfo("🗣 Speech Analysis", feedback)
        corrected = simpledialog.askstring("Edit Transcript", "Edit your response if needed:", initialvalue=transcript)
        final_text = corrected if corrected else transcript
        self.analyze_text(self.questions[self.current_q], final_text)

    def analyze_speech_rate_and_pauses(self, audio_file, transcript):
        word_count = len(transcript.strip().split())
        with contextlib.closing(wave.open(audio_file, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)

        wpm = round((word_count / duration) * 60, 2)
        pause_count = transcript.lower().count(" um") + transcript.lower().count(" uh") + transcript.lower().count("...")

        analysis = f"🗣 Speech Rate: {wpm} WPM\n⏸ Pauses detected: {pause_count}"
        comment = ""
        if wpm < 70:
            comment += "Your speech was slow. Be more fluid.\n"
        elif wpm > 160:
            comment += "You spoke fast. Slow down for clarity.\n"
        if pause_count > 3:
            comment += "Try to reduce filler words like 'um' or 'uh'."
        return f"{analysis}\n{comment.strip()}"

    def transcribe_google_file(self, filepath):
        with sr.AudioFile(filepath) as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = self.recognizer.record(source)
            try:
                return self.recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                return "[Unrecognized Speech]"
            except sr.RequestError:
                return "[Google API Error]"

    def manual_input(self):
        answer = simpledialog.askstring("Manual Answer", "Type your answer below:")
        if answer:
            matches = []
            if self.grammar_tool:
                try:
                    matches = self.grammar_tool.check(answer)
                except Exception:
                    matches = []
            if matches:
                issues = []
                for match in matches[:5]:
                    msg = getattr(match, "message", None)
                    issues.append(f"• {msg if msg else str(match)}")
                messagebox.showinfo("Grammar Issues", "\n".join(issues))
            else:
                messagebox.showinfo("Grammar Check", "No grammar issues found.")
            self.analyze_text(self.questions[self.current_q], answer)

    def analyze_text(self, question, answer):
        cleaned_answer = answer.strip()
        if not cleaned_answer:
            messagebox.showwarning("Empty Answer", "Please provide a valid answer.")
            return

        # Sentiment Analysis
        result = self.sentiment_model.predict(cleaned_answer)
        sentiment_label = result["label"].upper()
        sentiment_raw_score = result["confidence"]

        # Keyword Matching with Cosine Similarity
        answer_embedding = self.semantic_model.encode(cleaned_answer, convert_to_tensor=True)
        keyword_hits = 0
        expected_keyword_embeddings = self.keyword_embeddings.get(question, [])
        
        for kw_embedding in expected_keyword_embeddings:
            score = util.cos_sim(answer_embedding, kw_embedding).item()
            if score > 0.4:  # Similarity threshold
                keyword_hits += 1

        sentiment_score_0_5 = round(sentiment_raw_score * 5, 2)
        confidence = min(10.0, round((keyword_hits * 2) + sentiment_score_0_5, 2))

        self.transcript_label.config(text=f"Transcript: {cleaned_answer}")
        self.sentiment_label.config(
            text=f"Sentiment: {sentiment_label} ({round(sentiment_raw_score * 100, 1)}%)"
        )
        self.score_label.config(
            text=f"Confidence Score: {confidence}/10 | Keywords Matched: {keyword_hits}"
        )

        self.responses.append(
            {
                "question": question,
                "answer": cleaned_answer,
                "sentiment": sentiment_label,
                "sentiment_val": sentiment_score_0_5,
                "keywords_matched": keyword_hits,
                "confidence": confidence,
            }
        )

    def next_question(self):
        self.current_q += 1
        if self.current_q < len(self.questions):
            self.question_label.config(text=self.format_question())
            self.transcript_label.config(text="Transcript:")
            self.sentiment_label.config(text="Sentiment:")
            self.score_label.config(text="Confidence Score:")
        else:
            self.show_summary_popup()

    def show_summary_popup(self):
        total = len(self.responses)
        if total == 0:
            return
        avg_conf = sum(r['confidence'] for r in self.responses) / total
        avg_keys = sum(r['keywords_matched'] for r in self.responses) / total
        sentiments = [r['sentiment'] for r in self.responses]
        pos = sum(1 for s in sentiments if "POSITIVE" in s.upper())
        neg = sum(1 for s in sentiments if "NEGATIVE" in s.upper())
        neu = total - pos - neg

        summary = "🧠 Interview Summary\n\n"
        if avg_conf >= 7:
            summary += "✅ You were confident in most answers.\n"
        if avg_keys >= 2:
            summary += "✅ You stayed relevant and structured.\n"
        if pos > neu:
            summary += "✅ Your tone was positive and engaging.\n"
        if avg_conf < 6:
            summary += "⚠ Try to speak with more confidence.\n"
        if avg_keys < 2:
            summary += "⚠ Include more relevant examples.\n"
        if neu >= total / 2:
            summary += "⚠ Try to sound more enthusiastic.\n"

        messagebox.showinfo("💡 Strengths & Weaknesses", summary.strip())
    
    def show_graph(self, save_path=None):
        if not self.responses:
            messagebox.showwarning("No Data", "Please answer at least one question first.")
            return

        questions = [f"Q{i+1}" for i in range(len(self.responses))]
        confidences = [resp["confidence"] for resp in self.responses]
        sentiments = [resp["sentiment_val"] for resp in self.responses]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(questions, confidences, label="Confidence Score", color='skyblue')
        ax.plot(questions, sentiments, label="Sentiment Score (0–5)", color='orange', marker='o')
        ax.set_ylabel("Score")
        ax.set_title("Interview NLP Analysis")
        ax.legend()
        ax.set_ylim(0, 10)

        if save_path:
            fig.savefig(save_path)
            plt.close(fig)
        else:
            top = tk.Toplevel(self.root)
            top.title("NLP Score Chart")
            canvas = FigureCanvasTkAgg(fig, master=top)
            canvas.draw()
            canvas.get_tk_widget().pack()

    def generate_pdf(self):
        self.show_graph("graph.png")

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, "AI Interview Report", ln=True, align="C")
        pdf.ln(10)

        for i, res in enumerate(self.responses):
            pdf.set_font("Arial", size=11)
            pdf.multi_cell(0, 10, f"Q{i+1}: {res['question']}")
            pdf.multi_cell(0, 10, f"Answer: {res['answer']}")
            pdf.multi_cell(0, 10, f"Sentiment: {res['sentiment']}")
            pdf.multi_cell(0, 10, f"Keywords Matched: {res['keywords_matched']}")
            pdf.multi_cell(0, 10, f"Confidence Score: {res['confidence']}/10")
            pdf.ln(5)

        if os.path.exists("graph.png"):
            pdf.image("graph.png", x=10, y=pdf.get_y(), w=180)

        pdf.output("Interview_Report.pdf")
        messagebox.showinfo("📄 PDF Created", "Interview_Report.pdf has been generated!")


if __name__ == "__main__":
    root = tk.Tk()
    app = InterviewApp(root)
    root.mainloop()
