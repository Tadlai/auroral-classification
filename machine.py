import os
import joblib


class Machine:
    def __init__(self, data_dir=None, save_name=None):
        self.data_dir = data_dir
        self.save_name = save_name
        self._name = "Machine"

    def get_name(self):
        return self._name

    def get_params(self):
        dic = self.__dict__
        ret_dic = {}
        for key, value in dic.items():
            t = type(value)
            if t == str or t == tuple or t == float or t == int or t == bool or t == dict or value is None:
                ret_dic[key] = value
            else:  # convert types which are incompatible with joblib dump
                ret_dic[key] = type(value).__name__
        return ret_dic

    def verify_machine_dir(self, subdir):
        list_machines = os.listdir(subdir)
        save_name = self.get_name() + str(len(list_machines) + 1)
        for filename in list_machines:
            filepath = os.path.join(subdir, filename)
            if self.is_equal(filepath):
                save_name = filename
                break
        return save_name

    def save_machine(self, dir):
        if not os.path.isdir(dir):
            os.mkdir(dir)
        joblib.dump(self.get_params(), os.path.join(dir, "dict.txt"))

    def verify_and_save_machine(self, subdir):
        self.save_name = self.verify_machine_dir(subdir)
        new_dir = os.path.join(subdir, self.save_name)
        self.save_machine(new_dir)

    def is_equal(self, saved_machine):
        saved_dict = joblib.load(os.path.join(saved_machine, "dict.txt"))
        current_dict = self.get_params()
        del saved_dict['save_name'], current_dict['save_name'], saved_dict['data_dir'], current_dict['data_dir']
        if saved_dict == current_dict:
            return True
        else:
            return False

    def verify_data_dir(self, root_dir):
        list_dirs = os.listdir(root_dir)
        if self.save_name is not None:
            if self.save_name not in list_dirs:
                self.data_dir = os.path.join(root_dir, self.save_name)
                os.mkdir(self.data_dir)
            else:
                self.data_dir = os.path.join(root_dir, self.save_name)
                print("Data dir " + self.save_name + " already exists.")
        else:
            print("The machine must be saved before its data!")

    def save_data(self):
        print("Function save_data must be overridden!")

    def verify_and_save_data(self, root_dir):
        self.verify_data_dir(root_dir)
        self.save_data()
