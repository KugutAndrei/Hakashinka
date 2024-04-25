import numpy as np
# import sumo as sm
import os
from traffic_initialization import *
import sys
from bs4 import BeautifulSoup


class Simulation:
    def __init__(self, start_road, end_road, 
                 N=100, dur=100, sim_speed=13, default_gap=4, default_speed=20, work_dir='./xml_data/test', 
                 net_data_dir='./xml_data/net_data', net_data='narrows.net.xml'):
        """
        Class providing easier access to traffic simulation. Also should work for Windows
        users, but havent been tested yet
        
        :param start_road:
            str or list(str) 
            IDs of the vehicle spawn roads. For example ``E2``
        :param end_road:
            str or list(str) 
            IDs of the vehicle despawn roads. For example ``E3``
        :param N: (optinal)
            int
            Number of vehicles to spawn (default: ``100``)
        :param dur: (optinal)
            int
            Simulation duration (default: ``100``)
        :param sim_speed: (optinal)
            int
            Speed of the simulation (default: ``13``)
        :param default_gap: (optinal)
            float
            Gap between newborn vehicles (default: ``4``)
        :param default_speed: (optinal)
            float
            Maximal speed of a newborm vehicle (default: ``20``)
        :param work_dir: (optinal)
            str
            Directory to save all the simulation files (default: ``./xml_data/test``)
        :param net_data_dir: (optinal)
            str
            Directory to obtain .net.xml file for simulation (default: ``./xml_data/net_data``)
        :param net_data: (optinal)
            str
            Name of the .net.xml file for simulation (default: ``narrows.net.xml``)
        """

        self.work_dir = work_dir
        self.net_data_dir = net_data_dir
        self.netfile = net_data
        self.name = net_data.split('.')[0]
        self.routefile = f'{self.name}.rou.xml'
        self.configfile = f'{self.name}.sumocfg'
        self.outpufile = f'{self.name}.statistic.output.xml'
        self.addfile = f'{self.name}.add.xml'

        self.default_gap = default_gap
        self.default_speed = default_speed
        self.start_road = start_road
        self.end_road = end_road
        self.N = N
        self.dur = dur
        self.sim_speed = sim_speed

        # creating route class and additional data class
        self.rtFl = RouteFile(os.path.join(work_dir, self.routefile))
        self.addFl = AditionalFile(os.path.join(work_dir, self.addfile))

        self.default_vtype_dict = {'accel' : "3.0",
                                    'decel' : "6.0",
                                    'length' : "4.0",
                                    'minGap' : str(default_gap),
                                    'maxSpeed' : str(default_speed),
                                    'sigma' : "0.7"}

        # setting flow with default function
        self.set_flow(start_road, end_road, N, dur, sim_speed)

    def set_flow(self, start_road, end_road, N, dur, sim_speed):
        # auxilary metod
        self.rtFl.setVehicleType('default', self.default_vtype_dict.copy())
        if type(end_road) == str:
            end_road = [end_road]
        if type(start_road) == str:
            start_road = [start_road]
        for end_road_ in end_road:
            for start_road_ in start_road:
                self.rtFl.setVehicleFlow('default', source=start_road_, dest=end_road_, num=N, dur=dur, speed=sim_speed)

    def set_calibrator(self, road, lane=None, gap=None, speed=None):
        """
        Method for setting calibrators on the road in order to change vehicle speed or gap

        :param road: (
            str
            ID if the road to set calibrator 
        :param lane: (optinal)
            str, list(str) or None
            Lane(s) number(s) to set calibrator. If ``None`` sets calibrator to all
            avalible lanes. (default: ``None``)
        :param gap: (optinal)
            float
            Gap between vehicles on the road to set via calibrator 
            (default: ``default_gap`` in ``__init__`` metod)
        :param speed: (optinal)
            float
            Vehicles max speed on the road to set via calibrator 
            (default: ``default_speed`` in ``__init__`` metod)        
        """

        if gap is None:
            gap = self.default_gap
        if speed is None:
            speed = self.default_speed
        
        if lane is None:
            with open(os.path.join(self.net_data_dir, self.netfile), 'r') as file:
                data = file.read()
            data = BeautifulSoup(data, "xml")

            for edge in data('edge'):
                if edge['id'] == road:
                    lane_count = len(edge('lane'))
                    break
            else:
                raise Exception('Invalid road id!')

            vtype_dict = self.default_vtype_dict.copy()
            vtype_dict['minGap'] = str(gap)
            vtype_dict['maxSpeed'] = str(speed)
            self.addFl.setVehicleType(f'vtype_gap_{gap}_speed_{speed}', vtype_dict)
            for i in range(lane_count):
                self.addFl.setCalibrator(f'calib_gap_{gap}_speed_{speed}_road_{road}_lane{str(i)}', road, str(i))
                self.addFl.setVehicleFlow(f'calib_gap_{gap}_speed_{speed}_road_{road}_lane{str(i)}', f'vtype_gap_{gap}_speed_{speed}', "999999")


        elif type(lane) is str:
            vtype_dict = self.default_vtype_dict.copy()
            vtype_dict['minGap'] = str(gap)
            vtype_dict['maxSpeed'] = str(speed)
            self.addFl.setVehicleType(f'vtype_gap_{gap}_speed_{speed}', vtype_dict)
            self.addFl.setCalibrator(f'calib_gap_{gap}_speed_{speed}_road_{road}_lane{lane}', road, lane)
            self.addFl.setVehicleFlow(f'calib_gap_{gap}_speed_{speed}_road_{road}_lane{lane}', f'vtype_gap_{gap}_speed_{speed}', "999999")

        elif type(lane) is list:
            vtype_dict = self.default_vtype_dict.copy()
            vtype_dict['minGap'] = str(gap)
            vtype_dict['maxSpeed'] = str(speed)
            self.addFl.setVehicleType(f'vtype_gap_{gap}_speed_{speed}', vtype_dict)
            for lane_ in lane:
                self.addFl.setCalibrator(f'calib_gap_{gap}_speed_{speed}_road_{road}_lane{lane_}', road, lane_)
                self.addFl.setVehicleFlow(f'calib_gap_{gap}_speed_{speed}_road_{road}_lane{lane_}', f'vtype_gap_{gap}_speed_{speed}', "999999")

        else:
            raise Exception("Undefined lane type!")

    def generate_config(self):
        # auxilary metod
        try:
            self.rtFl.save()
            self.addFl.save()
        except FileNotFoundError:
            os.system(f'mkdir {self.work_dir}')
            self.rtFl.save()
            self.addFl.save()
        if os.name=='posix':
            os.system(f'cp {os.path.join(self.net_data_dir, self.netfile)} {self.work_dir}')
        else:
            # this havent been tested
            os.system(f'copy {os.path.join(self.net_data_dir, self.netfile)} {self.work_dir}')
        generateConfigFile(
            os.path.join(self.work_dir, self.configfile), 
            self.netfile,
            self.routefile,
            self.addfile,
            output=['statistic']
        )

    def run(self, seed=None):
        """
        Metod to run the simulation
        """
        self.generate_config()
        loadConfig(os.path.join(self.work_dir, self.configfile), seed=seed)

    def get_simtime(self):
        """
        Metod to obtain the simulation duration

        :return:
            float, which is simulation duration in conventional units
        """

        with open(os.path.join(self.work_dir, self.outpufile), 'r') as file:
            data = file.read()
        data = BeautifulSoup(data, "xml")
        return float(data('performance')[0]['duration'])
    
    def get_mean_throughput(self):
        """
        Metod to obtain the mean vehicle throughput calculated as ``N/t``
        where ``t`` is simulation duration

        :return:
            float, which is mean vehicle throughput in conventional units
        """

        t = self.get_simtime()
        return self.N/t
    
    def get_safety_features(self):
        """
        Method to obtain the number of collisions, emergency stops and emergency braking

        :return:
            dict
        """

        with open(os.path.join(self.work_dir, self.outpufile), 'r') as file:
            data = file.read()
        data = BeautifulSoup(data, "xml")
        safety = data('safety')[0]
        return {
            'collisions': int(safety['collisions']),
            'emergencyStops': int(safety['emergencyStops']),
            'emergencyBraking': int(safety['emergencyBraking'])
        }

    def get_fitted_throughput(self):
        raise Exception('This feature doesnt work cause Andrey fucked it up!')