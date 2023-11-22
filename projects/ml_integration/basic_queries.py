from surrealdb import SurrealDB


connection = SurrealDB("ws://localhost:8000/database/namespace")
# House Price Prediction-0.0.1

connection.signin({
    "username": "root",
    "password": "root",
})


outcome = connection.query("ml::Prediction<0.0.1>(1.0, 1.0,);")
print(outcome)

outcome = connection.query("SELECT * FROM ml::Prediction<0.0.1>(1.0, 1.0,);")
print(outcome)


if __name__ == "__main__":
    pass
