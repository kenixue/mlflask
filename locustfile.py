
from locust import HttpUser, User, task


class UserBehavior(User):
    def on_start(self):
        """ on_start is called when a Locust start before any task is scheduled """
        print('start')

    @task(1)
    def index(self):
        self.client.get("/")
        
    @task(2)
    def mlflask_predction(self):
        self.client.get("/prediction")

    


class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    min_wait = 1
    max_wait = 3


