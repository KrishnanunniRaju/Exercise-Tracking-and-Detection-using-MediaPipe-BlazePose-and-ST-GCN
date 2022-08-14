from src.ExerciseTracker import ExerciseTracker

if __name__ == '__main__':
    instance=ExerciseTracker(False,strategy='spatial',edge_importance=True)
    instance.track()