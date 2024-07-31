movies = [
    {"title": "Inception", "genre": "Sci-Fi"},
    {"title": "The Godfather", "genre": "Crime"},
    {"title": "Pulp Fiction", "genre": "Crime"},
    {"title": "Interstellar", "genre": "Sci-Fi"},
    {"title": "The Dark Knight", "genre": "Action"},
    {"title": "Fight Club", "genre": "Drama"},
    {"title": "Forrest Gump", "genre": "Drama"},
]
def recommend_movies(favorite_genre):
    recommendations = [movie["title"] for movie in movies if movie["genre"] == favorite_genre]
    return recommendations

favorite_genre = input("Enter your favorite genre: ")
recommendations = recommend_movies(favorite_genre)
if recommendations:
    print("We recommend the following movies:")
    for movie in recommendations:
        print(f"- {movie}")
else:
    print("Sorry, we have no recommendations for that genre.")
