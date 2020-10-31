gcloud projects describe $GOOGLE_CLOUD_PROJECT


gcloud config set project $GOOGLE_CLOUD_PROJECT

gcloud app create

gcloud app deploy

y


source ~/.mlflask/bin/activate



make install

#run docker

export PROJECT_ID=mlflask

docker build -t gcr.io/mlflask/app:v1 .

docker images


docker run --rm -p 8080:8080 gcr.io/mlflask/app:v1

#upload Docker to Container Registry


gcloud auth configure-docker

docker push gcr.io/mlflask/app:v1



#K8s Deploy

#Create GKE Cluster

gcloud config set compute/zone us-central1-a

gcloud container clusters create mlflask-cluster --num-nodes=3

gcloud container clusters get-credentials mlflask-cluster

gcloud compute instances list

kubectl create deployment mlflask-server --image=gcr.io/mlflask/app:v1




kubectl scale deployment mlflask-server --replicas=3

kubectl autoscale deployment mlflask-server --cpu-percent=80 --min=1 --max=5

kubectl get pods

kubectl expose deployment mlflask-server --type=LoadBalancer --port 80 --target-port 8080


kubectl get service


# http://[EXTERNAL-IP]:80


#gcloud container clusters delete [CLUSTER-NAME]


#locust -f locustfile.py


http://0.0.0.0:8080/
