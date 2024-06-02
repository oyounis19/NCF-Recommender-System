from pydantic import BaseModel, Field
from enum import Enum

class GenreEnum(str, Enum):
    Action = "Action"
    Adventure = "Adventure"
    Animation = "Animation"
    Childrens = "Children"
    Comedy = "Comedy"
    Crime = "Crime"
    Documentary = "Documentary"
    Drama = "Drama"
    Fantasy = "Fantasy"
    FilmNoir = "Film-Noir"
    Horror = "Horror"
    Musical = "Musical"
    Mystery = "Mystery"
    Romance = "Romance"
    SciFi = "Sci-Fi"
    Thriller = "Thriller"
    War = "War"
    Western = "Western"

class OccupationEnum(str, Enum):
    other = "other"
    educator = "educator"
    artist = "artist"
    clerical = "clerical"
    grad_student = "grad student"
    customer_service = "customer service"
    doctor = "doctor"
    executive = "executive"
    farmer = "farmer"
    homemaker = "homemaker"
    K_12_student = "K-12 student"
    lawyer = "lawyer"
    programmer = "programmer"
    retired = "retired"
    sales = "sales"
    scientist = "scientist"
    self_employed = "self-employed"
    engineer = "engineer"
    craftsman = "craftsman"
    unemployed = "unemployed"
    writer = "writer"

class Request(BaseModel):
    top_k: int = Field(10, ge=1, le=20, description="Number of recommendations")
    id: int = Field(..., ge=1, description="User\'s ID")
    age: int = Field(None, ge=1, le=99, description="User\'s age")
    occupation: OccupationEnum = Field(None, description="User\'s occupation")
    gender: str = Field(None, pattern="^(M|F)$", description="User\'s gender")
    genres: list[GenreEnum] = Field(None, min_items=3, max_items=5, description="User\'s favorite genres")
