from hypothesis import settings

settings.register_profile("ci", max_examples=40, deadline=None)
settings.load_profile("ci")
