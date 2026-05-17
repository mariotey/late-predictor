import ast
from haversine import haversine, Unit

def get_distance_km(row, origin_col, destination_col):
    try:
        origin = ast.literal_eval(row[origin_col]) if isinstance(row[origin_col], str) else row[origin_col]
        destination = ast.literal_eval(row[destination_col]) if isinstance(row[destination_col], str) else row[destination_col]
        return haversine(origin, destination, unit=Unit.KILOMETERS)

    except Exception as e:
        print.warning(f"{e}: Distance calc failed for: \n {row}")
        return None