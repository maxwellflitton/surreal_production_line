from surrealdb import SurrealDB


connection = SurrealDB("ws://localhost:8000/database/namespace")
# House Price Prediction-0.0.1

connection.signin({
    "username": "root",
    "password": "root",
})


# connection.query("CREATE user:tobie SET name = 'Tobie';")
# connection.query("CREATE user:jaime SET name = 'Jaime';")
outcome = connection.query("ml::Prediction<0.0.1>(1.0, 1.0);")
print("here is the basic ml: ", outcome)

outcome = connection.query("ml::Prediction<0.0.1>({squarefoot: 500.0, num_floors: 1.0});")
print("here is the buffered ml: ", outcome)

# outcome = connection.query("SELECT * FROM user;")
# print("here is the basic select: ", outcome)
#
# outcome = connection.query("SELECT * FROM ml::Prediction<0.0.1>(1.0, 1.0,);")
# print("here is the select ML: ", outcome)

# outcome = connection.query("ml::Prediction<0.0.1>(1.0, 1.0,);")
# print(outcome)

if __name__ == "__main__":
    pass
