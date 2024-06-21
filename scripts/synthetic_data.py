import pandas as pd
import numpy as np

# Set parameters for synthetic data
num_users = 100
num_courses = 50
num_ratings = 500

# Generate synthetic user IDs
user_ids = np.random.randint(1, num_users + 1, num_ratings)

# Generate synthetic course IDs
course_ids = np.random.randint(1, num_courses + 1, num_ratings)

# Generate synthetic ratings
ratings = np.random.uniform(1, 5, num_ratings).round(2)

# Generate synthetic review counts
review_counts = np.random.randint(1, 100, num_ratings)

# Generate synthetic skills
skills = ["Skill" + str(i) for i in range(num_courses)]
skills = np.random.choice(skills, num_ratings)

# Generate synthetic levels
levels = ["Beginner", "Intermediate", "Advanced"]
levels = np.random.choice(levels, num_ratings)

# Generate synthetic certificate types
certificate_types = ["None", "Certificate"]
certificate_types = np.random.choice(certificate_types, num_ratings)

# Generate synthetic duration
durations = ["1-2 Weeks", "2-4 Weeks", "1-2 Months", "3-6 Months"]
durations = np.random.choice(durations, num_ratings)

# Generate synthetic credit eligibility
credit_eligibility = [True, False]
credit_eligibility = np.random.choice(credit_eligibility, num_ratings)

# Create a DataFrame
data = {
    "user_id": user_ids,
    "course_id": course_ids,
    "rating": ratings,
    "reviewcount": review_counts,
    "skills": skills,
    "level": levels,
    "certificatetype": certificate_types,
    "duration": durations,
    "crediteligibility": credit_eligibility,
}

df = pd.DataFrame(data)

# Save to CSV
csv_path = "synthetic_coursera.csv"
df.to_csv(csv_path, index=False)
print(f"Synthetic data saved to {csv_path}")
