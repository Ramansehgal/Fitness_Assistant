from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

class ExerciseFeedbackGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

        '''
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a professional fitness coach providing corrective feedback."
             "Give concise, technical feedback (<100 words)."
             "Use ONLY the provided expert guidelines."
             "If numeric ranges are missing, Say: Not specified."),
            ("human",
             "Exercise: {exercise}\n\n"
             "Expert Guidelines:\n{knowledge}\n\n"
             "Observed Motion Metrics:\n{metrics}\n\n"
             "Provide:\n"
             "1. What is incorrect\n"
             "2. Why it matters biomechanically\n"
             "3. How to correct it safely")
        ])'''

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a professional fitness coach providing corrective feedback."
             "Give concise, technical feedback (<200 words)."
             "Use ONLY the provided expert guidelines."
             "If numeric ranges are missing, Say: Not specified."),
            ("human",
             "Exercise: {exercise}\n"
             "Major impacted joints:{major_joint_summary}\n"
             "Per-joint statistics:{joint_statistics}\n"
             "Rules from biomechanics knowledge:{retrieved_knowledge}\n"
             "Task:\n"
             "1. Identify which joints show the highest variability and explain why\n"
             "2. If all joints are within safe range, explicitly say so\n"
             "3. Mention timestamps of max and min angles\n"
             "4. Provide at most 3 actionable cues\n"
             "5. Keep response under 120 words\n")
        ])

        self.exercise_timeline_feedback_prompt = ChatPromptTemplate.from_messages([
            ("system",
                """
                You are a biomechanics-aware exercise analysis assistant.

                Rules:
                - Use ONLY the provided knowledge and metrics.
                - Do NOT repeat knowledge text verbatim.
                - Do NOT hallucinate missing facts.
                - Keep output concise and structured.
                - Max 2 joints per timestamp.
                """
            ),
            (
                "human",
                """
                Exercise: {exercise}
                Segment Duration: T={t_start}s to T={t_end}s
                Biomechanics Knowledge:{retrieved_knowledge}
                Time-indexed Joint Metrics:{timeline_data}
                Instructions:
                For each timestamp:
                    - List impacted joints
                    - Show mean angle (degrees) and mean bone length
                    - State whether angle is within recommended range
                    - Briefly explain motion change from previous timestamp
                Use EXACT format:
                [At T=0]:
                    - JOINT: angle=X°, bone=Y → OK/OUT
                Reason: ...
                End with:
                Summary:
                    - Overall rep quality
                    - Injury risk (if any)
                    - Top 2 corrective cues
                """
            ),
        ])

        self.exercise_analysis_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert fitness biomechanics coach. "
                "Use only provided data and open-source biomechanics knowledge. "
                "Do not diagnose medical conditions."
            ),
            (
                "human",
                """
                You are given structured biomechanical analysis from a vision-based exercise system.
                TASKS:
                1. Provide 2 chunks and there relevance score out of 10.
                2. Determine SAFE angle limits for impacted joints using open-source biomechanics.
                3. Create a table rating exercise quality on FOUR parameters (0–100) and an overall score.
                4. Provide FOUR injury-prevention recommendations for the given exercise for single video and multi case for each segment
                5. Assess overall exercise health (Safe / Slightly aggressive / High injury risk).
                6. Final Rating (Excellent[>80] / Good[>65] / Satisfactory[>50] / Needs Improvement[>35] / Poor[<35]) based on Overal Score given in 2

                Use timestamps when referencing issues.
                Keep response under {words} words.

                STRUCTURED INPUT:
                {json_input}
                """
            )
        ])
        self.parser = StrOutputParser()
    
    def format_timeline_for_llm(timeline):
        lines = []
        for step in timeline:
            t = step["time"]
            joint_lines = []
            for j in step["joints"]:
                status = "OK" if j["in_range"] else "OUT"
                joint_lines.append(
                    f"{j['joint']} angle={j['mean_angle']}°, bone={j['mean_bone_length']} → {status}"
                )
            lines.append(f"T={t}: " + "; ".join(joint_lines))
        return "\n".join(lines)
    
    def format_knowledge_docs(docs):
        seen = set()
        cleaned = []

        for d in docs:
            key = (d.metadata.get("source_file"), d.metadata.get("page"))
            if key not in seen:
                seen.add(key)
                cleaned.append(d.page_content.strip())

        return "\n\n".join(cleaned)

    def generate_feedback(self, exercise, major_joint_summary, knowledge, metrics):
        chain = self.prompt | self.llm | self.parser
        return chain.invoke({
            "exercise": exercise,
            "major_joint_summary": major_joint_summary,
            "retrieved_knowledge": knowledge,
            "joint_statistics": metrics
        })
    
    def generate_exercise_timeline_feedback(self, exercise, t_start, t_end, knowledge, timeline_data):
        exercise_feedback_chain =  self.exercise_timeline_feedback_prompt | self.llm | self.parser
        return exercise_feedback_chain.invoke({
            "exercise": exercise,
            "t_start": t_start,
            "t_end": t_end,
            "retrieved_knowledge": knowledge,
            "timeline_data": timeline_data
        })
    
    def generate_exercise_analysis_feedback(self, llm_input_json, words):

        # IMPORTANT: create fresh instances
        llm = self.llm.__class__(**self.llm.model_kwargs)
        parser = self.parser.__class__()

        chain = self.exercise_analysis_prompt | llm | parser

        return chain.invoke({
            "json_input": llm_input_json,
            "words" : words
        })

'''
(
            RunnableParallel(
                {
                    "exercise": RunnablePassthrough(),
                    "t_start": RunnableLambda(lambda x: x["t_start"]),
                    "t_end": RunnableLambda(lambda x: x["t_end"]),
                    "timeline_data": RunnableLambda(
                        lambda x: self.format_timeline_for_llm(x["timeline"])
                    ),
                    "retrieved_knowledge": RunnableLambda(
                        lambda x: self.format_knowledge_docs(x["knowledge_docs"])
                    ),
                }
            )
            |

'''