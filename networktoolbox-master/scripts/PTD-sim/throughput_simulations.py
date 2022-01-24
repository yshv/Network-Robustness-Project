
import sys
sys.path.append('../')
import simulation_base


if __name__ == "__main__":
    hostname = "128.40.41.48"
    port = 7112
    simulation_base.main(collection="HTD-distance-revised",
                         query={"ILP-connections":{"$exists":True}}, hostname=hostname, port=port,
                         ILP=True, desc="Ruijie PTD throughput calc", fibre_num=16, route_function="FF-kSP",
                         heuristic=True)
    simulation_base.main(collection="prufer-random-distance-revised",
                         query={"ILP-connections": {"$exists": True}}, hostname=hostname, port=port,
                         ILP=True, desc="Ruijie PTD throughput calc", fibre_num=16, route_function="FF-kSP",
                         heuristic=True)
    simulation_base.main(collection="prufer-select-distance-revised",
                         query={"ILP-connections": {"$exists": True}}, hostname=hostname, port=port,
                         ILP=True, desc="Ruijie PTD throughput calc", fibre_num=16, route_function="FF-kSP",
                         heuristic=True)
    simulation_base.main(collection="vector-ga-distance-revised",
                         query={"ILP-connections": {"$exists": True}}, hostname=hostname, port=port,
                         ILP=True, desc="Ruijie PTD throughput calc", fibre_num=16, route_function="FF-kSP",
                         heuristic=True)
    simulation_base.main(collection="prufer-select-ga-distance-revised",
                         query={"ILP-connections": {"$exists": True}}, hostname=hostname, port=port,
                         ILP=True, desc="Ruijie PTD throughput calc", fibre_num=16, route_function="FF-kSP",
                         heuristic=True)