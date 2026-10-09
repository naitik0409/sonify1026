"""
Sonify 1026 Voice Engine - Final Test
English & Hindi Only (No Urdu)
"""

import os
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from snfy import SonifyEngine, Emotion, Language


def main():
    engine = SonifyEngine()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/final_test_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    results_log = []
    total_tests = 0
    passed_tests = 0

    def run_test(name, text, emotion, language=None, accent="in"):
        nonlocal total_tests, passed_tests
        total_tests += 1
        print(f"\n[{total_tests}] {name}")
        try:
            print(f"    Text: {text[:80]}...")
        except UnicodeEncodeError:
            print(f"    Text: [Hindi text - {len(text)} chars]")
        print(f"    Emotion: {emotion.value}")

        try:
            start = time.time()
            result = engine.synthesize(
                text=text,
                emotion=emotion,
                language=language,
                output_path=os.path.join(output_dir, f"{total_tests:03d}_{name.replace(' ', '_')}.wav"),
                accent=accent,
            )
            elapsed = time.time() - start

            log_entry = {
                "test_number": total_tests,
                "name": name,
                "text": text,
                "emotion": emotion.value,
                "duration_seconds": round(result.duration, 3),
                "status": "PASSED"
            }
            results_log.append(log_entry)
            passed_tests += 1
            print(f"    PASSED ({elapsed:.2f}s, {result.duration:.2f}s audio)")
        except Exception as e:
            log_entry = {
                "test_number": total_tests,
                "name": name,
                "text": text,
                "emotion": emotion.value,
                "status": "FAILED",
                "error": str(e)
            }
            results_log.append(log_entry)
            print(f"    FAILED: {e}")

    print("=" * 70)
    print("Sonify 1026 - English & Hindi Only (No Urdu)")
    print("=" * 70)

    # ============================================================
    # English Tests
    # ============================================================
    print("\n" + "=" * 70)
    print("English Tests")
    print("=" * 70)

    english_tests = [
        ("eng_greeting_1", "Hello! How are you? I hope you are doing well.", Emotion.HAPPY),
        ("eng_greeting_2", "Good morning! Welcome to our service.", Emotion.WARM),
        ("eng_thanks_1", "Thank you so much! I really appreciate your help.", Emotion.HAPPY),
        ("eng_thanks_2", "Thanks a lot! You have been very kind.", Emotion.WARM),
        ("eng_farewell_1", "Goodbye! Take care of yourself. See you soon.", Emotion.WARM),
        ("eng_farewell_2", "Bye! Have a great day ahead.", Emotion.HAPPY),
        ("eng_agreement_1", "Okay, sure! I will be there on time.", Emotion.NEUTRAL),
        ("eng_agreement_2", "Yes, that sounds good. Let us do it.", Emotion.NEUTRAL),
        ("eng_praise_1", "Wow! That is amazing! You did a great job!", Emotion.EXCITED),
        ("eng_praise_2", "Excellent work! I am really impressed!", Emotion.HAPPY),
        ("eng_sympathy_1", "I am sorry to hear that. I hope things get better.", Emotion.SAD),
        ("eng_sympathy_2", "Do not worry, everything will be fine.", Emotion.REASSURING),
        ("eng_casual_1", "Hey, what is up? Long time no see!", Emotion.HAPPY),
        ("eng_casual_2", "Come on, let us go out. It is too boring here.", Emotion.PLAYFUL),
        ("eng_casual_3", "Sure, no problem. See you tomorrow.", Emotion.CALM),
        ("eng_question_1", "Where are you going? Do you need any help?", Emotion.NEUTRAL),
        ("eng_question_2", "What happened? Is everything okay?", Emotion.THOUGHTFUL),
        ("eng_worry_1", "I am feeling a bit unwell. I think I need to rest.", Emotion.TIRED),
        ("eng_worry_2", "I am scared. Please stay with me.", Emotion.FEARFUL),
        ("eng_news_1", "Did you hear the news? Something big happened!", Emotion.SURPRISED),
        ("eng_news_2", "That is interesting. Tell me more about it.", Emotion.THOUGHTFUL),
        ("eng_love_1", "I miss you so much. Life is not the same without you.", Emotion.SAD),
        ("eng_love_2", "You are the best. Thank you for everything.", Emotion.WARM),
        ("eng_work_1", "How is the work going? Any problems?", Emotion.NEUTRAL),
        ("eng_work_2", "Please submit the report by tomorrow.", Emotion.SERIOUS),
        ("eng_food_1", "The food is delicious! Thank you!", Emotion.HAPPY),
        ("eng_food_2", "Would you like some tea? It is very cold outside.", Emotion.CALM),
        ("eng_weather_1", "It is raining heavily. Stay at home.", Emotion.CALM),
        ("eng_weather_2", "It is very hot today. Drink plenty of water.", Emotion.TIRED),
        ("eng_study_1", "How are your studies going? Any difficulties?", Emotion.NEUTRAL),
        ("eng_study_2", "The exam is coming. Study hard.", Emotion.REASSURING),
        ("eng_travel_1", "When are you arriving? How long will it take?", Emotion.NEUTRAL),
        ("eng_travel_2", "The train has arrived. Hurry up!", Emotion.HAPPY),
        ("eng_health_1", "Take some rest. You look tired.", Emotion.CALM),
        ("eng_health_2", "Take this medicine. You will feel better.", Emotion.REASSURING),
    ]

    for name, text, emotion in english_tests:
        run_test(name, text, emotion, Language.EN_IN)

    # ============================================================
    # Hindi Tests (Pure Hindi, No Urdu)
    # ============================================================
    print("\n" + "=" * 70)
    print("Hindi Tests (Pure Hindi)")
    print("=" * 70)

    hindi_tests = [
        ("hin_greeting_1", "नमस्ते! आप कैसे हैं? सब ठीक है?", Emotion.HAPPY),
        ("hin_greeting_2", "सुप्रभात! आपका दिन शुभ हो।", Emotion.WARM),
        ("hin_thanks_1", "धन्यवाद! आपकी बहुत बहुत मेहरबानी।", Emotion.HAPPY),
        ("hin_thanks_2", "बहुत धन्यवाद! तुमने मुझे बहुत मदद की।", Emotion.WARM),
        ("hin_farewell_1", "अलविदा! फिर मिलेंगे।", Emotion.WARM),
        ("hin_farewell_2", "बाय! अपना ध्यान रखना।", Emotion.HAPPY),
        ("hin_agreement_1", "ठीक है! मैं आ जाऊँगा।", Emotion.NEUTRAL),
        ("hin_agreement_2", "हाँ, बिल्कुल सही कह रहे हो।", Emotion.NEUTRAL),
        ("hin_praise_1", "वाह! क्या बात है! शाबाश!", Emotion.EXCITED),
        ("hin_praise_2", "बहुत बढ़िया! तुमने बहुत अच्छा किया!", Emotion.HAPPY),
        ("hin_sympathy_1", "सुनकर बहुत दुख हुआ। उम्मीद है सब ठीक हो जाएगा।", Emotion.SAD),
        ("hin_sympathy_2", "चिंता मत करो, सब ठीक हो जाएगा।", Emotion.REASSURING),
        ("hin_casual_1", "अरे यार! क्या हाल है? बहुत दिन हो गए।", Emotion.HAPPY),
        ("hin_casual_2", "चलो बाहर चलते हैं। बहुत बैठे बैठे हैं।", Emotion.PLAYFUL),
        ("hin_casual_3", "ठीक है, कल मिलते हैं।", Emotion.CALM),
        ("hin_question_1", "तुम कहाँ जा रहे हो? कोई काम है?", Emotion.NEUTRAL),
        ("hin_question_2", "क्या हुआ? सब ठीक तो है ना?", Emotion.THOUGHTFUL),
        ("hin_worry_1", "मुझे बहुत थकान हो रही है। आराम करना है।", Emotion.TIRED),
        ("hin_worry_2", "डर लग रहा है। मुझे अकेले मत छोड़ो।", Emotion.FEARFUL),
        ("hin_news_1", "खबर सुनी? क्या हुआ वहाँ?", Emotion.SURPRISED),
        ("hin_news_2", "यह दिलचस्प है। और बताओ।", Emotion.THOUGHTFUL),
        ("hin_love_1", "तुम्हारे बिना सब अधूरा है।", Emotion.SAD),
        ("hin_love_2", "तुम सबसे अच्छे हो। धन्यवाद।", Emotion.WARM),
        ("hin_work_1", "काम कैसे चल रहा है? कोई समस्या है?", Emotion.NEUTRAL),
        ("hin_work_2", "रिपोर्ट कल तक देना है। ध्यान से।", Emotion.SERIOUS),
        ("hin_food_1", "खाना बहुत स्वादिष्ट है! धन्यवाद!", Emotion.HAPPY),
        ("hin_food_2", "चाय पिएंगे? बहुत ठंड हो रही है।", Emotion.CALM),
        ("hin_weather_1", "बहुत बारिश हो रही है। घर पे ही रहो।", Emotion.CALM),
        ("hin_weather_2", "बहुत गर्मी हो रही है। पानी पियो।", Emotion.TIRED),
        ("hin_study_1", "पढ़ाई कैसे चल रही है? कोई समस्या है?", Emotion.NEUTRAL),
        ("hin_study_2", "परीक्षा आ रही है। ध्यान से पढ़ो।", Emotion.REASSURING),
        ("hin_travel_1", "कब आ रहे हो? कितनी देर लगेगी?", Emotion.NEUTRAL),
        ("hin_travel_2", "ट्रेन आ गई। चलो पकड़ो।", Emotion.HAPPY),
        ("hin_health_1", "आराम करो। तुम्हें थकान हो गई है।", Emotion.CALM),
        ("hin_health_2", "यह दवा लो। तबीयत ठीक हो जाएगी।", Emotion.REASSURING),
    ]

    for name, text, emotion in hindi_tests:
        run_test(name, text, emotion, Language.HI_IN)

    # ============================================================
    # Save results log
    # ============================================================
    log_path = os.path.join(output_dir, "test_results.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "results": results_log
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()