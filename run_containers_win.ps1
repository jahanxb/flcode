# Define the path to your code directory
$codePath = "C:\Users\mkhan40\Documents\vscodeprojects\flcode"

# Loop to create and run 10 containers
for ($i = 1; $i -le 10; $i++) {
    $ip = "172.18.0." + ($i + 1)
    docker run -d --name "ubuntu_container_$i" --net mynetwork --ip $ip -v "$codePath:/usr/src/app" myubuntuimage
}
