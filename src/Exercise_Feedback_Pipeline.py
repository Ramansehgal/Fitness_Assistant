# from analysis.exercise_profiles import EXERCISE_PROFILES

import mediapipe as mp

from dotenv import load_dotenv
from motion_analyzer import MotionAnalyzer
from LLM_Feedback import ExerciseFeedbackGenerator
from Knowledge_Ingestor import ExerciseKnowledgeIngestor
from Knowledge_Retriever import ExerciseKnowledgeRetriever

from collections import defaultdict
from motion_analyzer import MotionAnalyzer
from feature_extractor import PoseFeatureExtractor

load_dotenv()  # expects OPENAI_API_KEY in .env

EXERCISE_PROFILES = {
    "squat": {
        "primary_joints": ["hip", "knee", "ankle", "spine"],
        "angles": ["knee_angle", "hip_angle", "spine_angle"],
        "symmetry_pairs": [("left_knee", "right_knee")]
    },
    "pushup": {
        "primary_joints": ["elbow", "shoulder", "spine"],
        "angles": ["elbow_angle", "spine_angle"]
    }
}

# One entry per frame (sampled at fixed FPS)

FRAME_METRICS = [
    {
        "time": 3.0,
        "knee_angle": 62.5,
        "hip_angle": 55.2,
        "spine_angle": 12.1,
        "left_knee": 63.1,
        "right_knee": 61.8
    },
    {
        "time": 3.1,
        "knee_angle": 64.0,
        "hip_angle": 57.0,
        "spine_angle": 11.8,
        "left_knee": 64.5,
        "right_knee": 63.2
    },
    {
        "time": 3.2,
        "knee_angle": 65.3,
        "hip_angle": 58.4,
        "spine_angle": 12.4,
        "left_knee": 66.0,
        "right_knee": 64.7
    }
]

SEGMENT_ANALYSIS = {
    "exercise": "squat",
    "segment_time": [3.0, 7.0],
    "metrics": {
        "knee_angle": {
            "knee_angle_at_each_segment_time" : [62.5, 63.4, 64.0, 64.7, 65.3],
            "mean": 64.0,
            "std": 1.1,
            "min": 62.5,
            "max": 65.3,
            "out_of_range_pct": 85.0
        },
        "hip_angle": {
            "mean": 56.9,
            "std": 1.6
        },
        "symmetry": {
            "knee_difference_mean": 1.3
        }
    }
}


class ExerciseFeedbackPipeline:
    def __init__(self, knowledge_retriever):
        self.knowledge_retriever = knowledge_retriever
        self.feedback_gen = ExerciseFeedbackGenerator()

    def run(self, exercise, segment_metrics):
        profile = EXERCISE_PROFILES[exercise]
        analyzer = MotionAnalyzer(profile)

        motion_summary = analyzer.analyze_segment(segment_metrics)

        knowledge = self.knowledge_retriever.retrieve(
            exercise,
            "recommended joint angles and common mistakes"
        )

        feedback = self.feedback_gen.generate_feedback(
            exercise,
            knowledge,
            motion_summary
        )

        return {
            "exercise": exercise,
            "analysis": motion_summary,
            "feedback": feedback
        }

#------------------------------------------------------------------------------------------------------

mp_pose = mp.solutions.pose

def describe_angle(self, angle_id):
    a, b, c = self.geometry.angle_id_map[angle_id]
    return {
        "angle_id": angle_id,
        "center_joint": b,
        "joint_triplet": (a, b, c),
        "center_joint_name": mp_pose.PoseLandmark(b).name
    }


def get_dominant_bone_pair(angle_triplet):
    """
    angle_triplet: (a, b, c) -> angle at b
    returns: (b, c) dominant bone pair
    """
    a, b, c = angle_triplet
    return (b, c)

def build_bone_dublets():
    return 

def build_angle_triplets(mp_pose, exclude_centers=None):
    
    if exclude_centers is None:
        exclude_centers = []

    # Undirected adjacency from POSE_CONNECTIONS
    adj = defaultdict(set)
    for a, c in mp_pose.POSE_CONNECTIONS:
        adj[a].add(c)
        adj[c].add(a)

    triplets = []
    for b, neighbors in adj.items():
        if b in exclude_centers:
            continue
        neighbors = list(neighbors)
        if len(neighbors) < 2:
            continue
        n = len(neighbors)
        for i in range(n):
            for j in range(i+1, n):
                a = neighbors[i]
                c = neighbors[j]
                triplets.append((a, b, c))
    return triplets

# --- 1. Feature extractor ---
feature_extractor = PoseFeatureExtractor(sample_fps=10, max_frames=None, exclude_angle_centers=None)

# --- 2. Angle & bone definitions ---
# angle_id -> (a, b, c)
EXCLUDED_CENTERS = list(range(0, 11)) + list(range(15, 23))  
ANGLE_TRIPLETS = build_angle_triplets(mp_pose, exclude_centers=EXCLUDED_CENTERS)

angle_id_map = {i: triplet for i, triplet in enumerate(ANGLE_TRIPLETS)}

# dominant bone per angle (chosen earlier)
# dominant bone = (center_joint, distal_joint)
dominant_bone_map = {
    i: (triplet[1], triplet[2])   # (b, c)
    for i, triplet in angle_id_map.items()
}

# --- 3. Segment analyzer ---
motion_analyzer = MotionAnalyzer(
    feature_extractor=feature_extractor,
    angle_triplets=angle_id_map,
    dominant_bone_map=dominant_bone_map,
    sample_fps=10
)

video_path = "data/videos/squat.mp4"

frames_analysis_per_sec = motion_analyzer.build_frames_timestamp_per_video(video_path)

print(f"Total frames extracted: {len(frames_analysis_per_sec)}")
print("Sample frame:\n", frames_analysis_per_sec[0])

VALID_ANGLE_RANGES = {
    "knee": (70, 110),
    "hip": (80, 130),
    "elbow": (150, 180)
}

SEGMENT_ANALYSIS_2 = motion_analyzer.analyze_segment(frames_analysis_per_sec, VALID_ANGLE_RANGES)
SEGMENT_ANALYSIS_2["exercise"] = "squat"
from pprint import pprint
pprint(SEGMENT_ANALYSIS_2)

SEGMENT_ANALYSIS_2_FORMATED = motion_analyzer.format_segment_for_llm(SEGMENT_ANALYSIS_2)
from pprint import pprint
pprint(SEGMENT_ANALYSIS_2_FORMATED)

#--------------------------------------------------------------------------------------------------------


# 1) Ingest multiple PDFs
ingestor = ExerciseKnowledgeIngestor(
    persist_dir="data/vectorstore"
)

vectorstore = ingestor.ingest_pdf(
    "data/knowledge_documents/squat.pdf",
    exercise="squat",
    joint="knee",
    source="biomechanics_notes"
)

vectorstore = ingestor.ingest_pdf(
    "data/knowledge_documents/joint.pdf",
    exercise="general",
    joint="general",
    source="safety_guidelines"
)
'''
vectorstore = ingestor.ingest_pdfs([
    "data/knowledge_documents/squat.pdf",
    "data/knowledge_documents/joint.pdf"
])
'''
print("Total docs in vectorstore:", len(vectorstore.index_to_docstore_id))
'''
docstore = vectorstore.docstore._dict
for i, doc in list(docstore.items())[:5]:
    print(doc.metadata)
'''
# 2) Create retriever
retriever = ExerciseKnowledgeRetriever(vectorstore)

# 3) Query
answer = retriever.retrieve(
    exercise="squat",
    question="What is the recommended knee angle range at the bottom position?"
)

print("\n------------------------------------- RAG Answer -------------------------------------")
print(answer)
print("------------------------- Provided Relevant Answer for sample query -----------------------------")
# retriever = ExerciseKnowledgeRetriever(vectorstore)
feedback_gen = ExerciseFeedbackGenerator()

knowledge = retriever.retrieve_with_debug2(
    exercise="squat",
    joint="knee",
    question="recommended knee angle and common mistakes"
)
feedback = ""

feedback = feedback_gen.generate_feedback(
    exercise="squat",
    major_joint_summary = SEGMENT_ANALYSIS_2_FORMATED,
    knowledge=knowledge,
    metrics=SEGMENT_ANALYSIS_2
)

print("\n---------------------------------- RAG Feedback -------------------------------------")
print(f"Knowledge: \n{knowledge}\nFeedback: \n{feedback}")
print("--------------------------- Provided Relevant Feedback for Sample Query -------------------------")
timeline_data = motion_analyzer.build_llm_timeline_input(SEGMENT_ANALYSIS_2_FORMATED,[0,7],VALID_ANGLE_RANGES)

timestamp_analysis = feedback_gen.generate_exercise_timeline_feedback(
    exercise="squat", t_start=0 ,t_end=7,
    knowledge=knowledge,
    timeline_data=timeline_data
)

print("\n---------------------------------- timestamp_analysis-------------------------------------")
print(f"Knowledge: \n{knowledge}\nFeedback: \n{timestamp_analysis}")
print("--------------------------- timestamp_analysis completed -------------------------------")

has_query = True
need_feedback = True
while(has_query == True):
    user_question = input("Please specifiy the fitness Query (STOP to Quit):\n")
    if user_question == "STOP":
        has_query = False
        continue
    answer = retriever.retrieve(
        exercise="squat",
        question=user_question
    )
    print(f"Q: {user_question}")
    print(f"A: {answer}")

while(need_feedback == True):
    user_question = input("Please specifiy the fitness query for feedback (STOP to Quit):\n")
    if user_question == "STOP":
        need_feedback = False
        continue
    knowledge = retriever.retrieve_with_debug2(
        exercise="squat",
        joint="knee",
        question=user_question
    )

    feedback = feedback_gen.generate_feedback(
        exercise="squat",
        knowledge=knowledge,
        metrics=SEGMENT_ANALYSIS
    )
    print(f"Q: {user_question}")
    print(f"Knowledge: \n{knowledge}\nFeedback: \n{feedback}")


'''

Sehgal
Vijay
Kumar

headvks@gmail.com
+91-9415947525

House no. 109, Kailash Residency, Mahakali Vidhyap
Near Bus Stand, Jhansi U.P.
Jhansi 284002
Uttar Pradesh

Academic Mentor
From 1st April 2025
Mathematics and Computer Application

Bundelkhand University
Kanpur Road, Bundelkhand University, Jhansi, Uttar
Jhansi 284128
Uttar Pradesh

Ph.D
Prof Sudhendu Biswas
Study of Stochastic Process Oriented Model of Human Reproductive
Delhi University, Dept of Mathematical Statistics

Queuing Theory, Exploratory Data Analysis
Biomedical Statistics, Big Data
Survival Analysis

I have served as a Professor of Statistics since 1999, and throughout my academic career, I have actively integrated statistical theory with computational methods and real-world applications. My areas of specialization includes Statistical Inference, Bayesian Analysis, Survival Analysis, Optimization, Reliability Theory, Biostatistics, Data Science, Big Data and Numerical Computation of Stochastic Models which align closely with the growing demands of data science education and research.

gurpritgrover@yahoo.com
Dr. Gurprit Grover
Ex Professor of Statistics, Delhi University, Delhi

pkkapur1@gmail.com
Prof P K Kapur
Director Interdiscipline research center, Amity University, Noida

rksonidmc@yahoo.com
Prof RK Soni
Professor of Biostatistics, Department of Community Medicine, Dayanand Medical College and Hospital Ludhiana Punjab



Introduction
6200664427
'''