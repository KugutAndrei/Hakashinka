import numpy as np


def make_cars_xml(inflow: np.array, cars_file_xml: str="cars.xml") -> None:
    """
    Make cars.xml file using inflow

    Args:
        inflow (arraylike): поток машин в каждой полосе
        cars_file_xml (str): имя файла для записи машин

    Returns:
        None | str: может возвращать файл.xml
    """
    pass


def make_simulation(dist: float, road: str, inflow: np.array) -> None | str:
    """
    Make simulation of cars

    Args:
        dist (float): минимальное расстояние между машинами
        road (string): название файла дороги для симуляции "road_name.xml"
        inflow (arraylike):  поток машин в каждой полосе

    Returns:
        None | str: может вернуть файл(ы), полученный после симуляции
    """
    pass


def loss_func(dist: float, road: str, inflow) -> float:
    """
    Calculate loss functction
    Фактически запускает создает файл для симуляции движения, запускает make_simulation()
    и считает по выходному файлу что-то

    Args:
        dist (float): минимальное расстояние между машинами
        road (string): название файла дороги для симуляции "road_name.xml"
        inflow (arraylike):  поток машин в каждой полосе

    Returns:
        float: что-то
    """
    return (dist - 1.0)**2 + 1.0




def main():
    print("main() from Loss.py")
    return 0

if __name__ == "__main__":
    main()
