
from locust import HttpUser, User, task


class UserBehavior(User):
    def on_start(self):
        """ on_start is called when a Locust start before any task is scheduled """
        print('start')

    @task
    def index(self):
        self.client.get("/")

    


class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    min_wait = 1
    max_wait = 3

if __name__ == "__main__":
    import os
    os.system("locust -f locustfile.py")