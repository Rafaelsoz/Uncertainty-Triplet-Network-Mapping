class MetricsLogger:
    def __init__(self, prefix=None):
        self.prefix = prefix

        self.metrics = []
        self.log_dict = {}


    def apply_prefix(self, name) -> str:
        return f'{self.prefix}_{name}'
    

    def add_metric(self, name):
        metric_name = self.apply_prefix(name)
        self.metrics.append(metric_name)
        self.log_dict[metric_name] = []


    def add_metrics(self, metrics: list):
        for metric_name in metrics:
            metric_name = self.apply_prefix(metric_name)
            self.metrics.append(metric_name)
            self.log_dict[metric_name] = []


    def update(self, stats: dict):
        for name, value in stats.items():
            metric_name = self.apply_prefix(name)

            if metric_name in self.metrics:
                self.log_dict[metric_name].append(value)


    def get_metrics(self) -> dict:
        return self.log_dict


    def get_metric(self, name: str) -> list:
        return self.log_dict[name]