class CraverDesignEnv(gym.Env):
    """
    For the higher level agent, modifies the net file based on the design decision.
    """
    def __init__(self, args, worker_id=None):
        super().__init__()
        self.args = args
        self.worker_id = worker_id
        self.unique_suffix = f"_{worker_id}" if worker_id is not None else ""
        self.original_net_file = './SUMO_files/original_craver_road.net.xml'
        self.modified_net_file = self.original_net_file.replace('.xml', f'{self.unique_suffix}.xml')


