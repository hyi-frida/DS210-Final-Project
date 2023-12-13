use petgraph::graph::{Graph, NodeIndex};
use petgraph::algo::dijkstra;
use rand::seq::SliceRandom;
use std::collections::HashMap;

struct GraphAnalysis {
    graph: Graph<&'static str, &'static str>,
}

impl GraphAnalysis {
    fn new() -> Self {
        GraphAnalysis {
            graph: Graph::new(),
        }
    }

    // Add a simple method for manually creating graphs
    fn create_simple_graph(&mut self) {
        let node_a = self.graph.add_node("Node A");
        let node_b = self.graph.add_node("Node B");
        let node_c = self.graph.add_node("Node C");
        let node_d = self.graph.add_node("Node D");
        let node_e = self.graph.add_node("Node E");

        self.graph.add_edge(node_a, node_b, "Edge A-B");
        self.graph.add_edge(node_a, node_c, "Edge A-C");
        self.graph.add_edge(node_b, node_c, "Edge B-C");
        self.graph.add_edge(node_c, node_d, "Edge C-D");
        self.graph.add_edge(node_d, node_e, "Edge D-E");
    }

    fn load_data(&mut self) {
        // Implement data loading logic based on specific datasets
        // example：self.graph.add_edge(NodeIndex::new(0), NodeIndex::new(1), "EdgeLabel");
    }

    fn calculate_distance(&self, start: NodeIndex, end: NodeIndex) -> Option<usize> {
        dijkstra(&self.graph, start, Some(end), |_| 1)
            .get(&end)
            .cloned()
    }

    fn calculate_degree_distribution(&self) {
        for node in self.graph.node_indices() {
            let degree = self.graph.neighbors(node).count();
            println!("Node {:?} has degree: {}", node.index(), degree);
        }
    }

    fn friends_of_friends_similarity(&self, vertex1: NodeIndex, vertex2: NodeIndex) {
        // Get the set of neighbors for two vertices
        let neighbors1: Vec<NodeIndex> = self.graph.neighbors(vertex1).collect();
        let neighbors2: Vec<NodeIndex> = self.graph.neighbors(vertex2).collect();

        // Calculate the number of common neighbors between two vertices
        let common_neighbors = neighbors1
            .iter()
            .filter(|&&neighbor| neighbors2.contains(&neighbor))
            .count();

        // Calculate similarity measures
        let similarity = common_neighbors as f64 / (neighbors1.len() + neighbors2.len()) as f64;

        println!(
            "Similarity between {:?} and {:?}: {:.2}",
            vertex1.index(),
            vertex2.index(),
            similarity
        );
    }

    fn graph_clustering(&self, k: usize) {
        let nodes: Vec<NodeIndex> = self.graph.node_indices().collect();
        let mut rng = rand::thread_rng();

        // Randomly select k initial representative nodes
        let mut representatives: Vec<NodeIndex> = nodes.choose_multiple(&mut rng, k).cloned().collect();

        // Iterative updates represent nodes
        for _ in 0..10 {
            // Update the average value of representative nodes for each group to assign each node to the nearest representative node
            let assignments: Vec<usize> = nodes
                .iter()
                .map(|&node| {
                    representatives
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, &rep)| self.calculate_distance(node, rep).unwrap_or(usize::MAX))
                        .map(|(index, _)| index)
                        .unwrap()
                })
                .collect();

            // Update the representative node to the average value for each group
            for (rep_index, representative) in representatives.iter_mut().enumerate() {
                let group_nodes: Vec<NodeIndex> = nodes
                    .iter()
                    .zip(assignments.iter())
                    .filter(|&(_, &index)| index == rep_index)
                    .map(|(&node, _)| node)
                    .collect();

                if !group_nodes.is_empty() {
                    *representative = self.calculate_centroid(&group_nodes);
                }
            }
        }

        // Print the final representative node
        println!("Final representatives: {:?}", representatives);
    }

    fn find_next_node(&self, current_subgraph: &[NodeIndex]) -> Option<NodeIndex> {
        let mut neighbors_to_add: Vec<NodeIndex> = Vec::new();

        // Traverse each node in the current subgraph
        for &node in current_subgraph {
            // Find neighboring nodes of a node that are not yet included in the subgraph
            let neighbors: Vec<NodeIndex> = self.graph.neighbors(node).collect();
            let new_neighbors: Vec<NodeIndex> = neighbors
                .iter()
                .filter(|&&neighbor| !current_subgraph.contains(&neighbor))
                .cloned()
                .collect();

            // Add neighboring nodes that are not included to the list to be added
            neighbors_to_add.extend(new_neighbors);
        }

        // Select a neighboring node from the list to be added as the next node
        neighbors_to_add.pop()
    }

    fn calculate_subgraph_density(&self, subgraph: &[NodeIndex]) -> f64 {
        let num_edges: usize = subgraph
            .iter()
            .flat_map(|&node| self.graph.edges(node))
            .count();

        let num_nodes = subgraph.len();

        if num_nodes > 0 {
            num_edges as f64 / num_nodes as f64
        } else {
            0.0
        }
    }

    fn calculate_centroid(&self, nodes: &[NodeIndex]) -> NodeIndex {
        if nodes.is_empty() {
            // If the node set is empty, return a virtual node
            return NodeIndex::new(0);
        }

        // Calculate the total number of node degrees
        let total_degree: usize = nodes.iter().map(|&node| self.graph.neighbors(node).count()).sum();

        // Find the node with the highest degree
        let centroid = nodes.iter().max_by_key(|&&node| self.graph.neighbors(node).count()).unwrap();

        *centroid
    }

    fn densest_subgraph(&self) {
        let mut max_density = 0.0;
        let mut best_subgraph: Vec<NodeIndex> = Vec::new();

        // Traverse all nodes, with each node serving as the starting node for the subgraph
        for start_node in self.graph.node_indices() {
            let mut current_subgraph: Vec<NodeIndex> = vec![start_node];

            // Greedily add neighboring nodes until they can no longer be added
            while let Some(next_node) = self.find_next_node(&current_subgraph) {
                current_subgraph.push(next_node);
            }

            // Calculate the density of the current subgraph
            let density = self.calculate_subgraph_density(&current_subgraph);

            // Update subgraphs with maximum density
            if density > max_density {
                max_density = density;
                best_subgraph = current_subgraph.clone();
            }
        }

        // Print information on the densest subgraph
        println!("Densest Subgraph: {:?}", best_subgraph);
        println!("Density: {:.2}", max_density);
    }

    fn centrality_measures(&self) {
        // 计算接近度中心度
        self.calculate_closeness_centrality();

        // 计算介数中心度
        self.calculate_betweenness_centrality();
    }

    fn calculate_closeness_centrality(&self) {
        println!("Closeness Centrality:");

        for node in self.graph.node_indices() {
            let shortest_paths = dijkstra(&self.graph, node, None, |_| 1);

            // 检查是否存在不可达的节点
            if shortest_paths.values().any(|&distance| distance == usize::MAX) {
                println!("Node {}: Unreachable", node.index());
            } else {
                // 计算接近度中心度
                let closeness_centrality = 1.0 / shortest_paths.values().map(|&distance| distance as f64).sum::<f64>();
                println!("Node {}: {:.2}", node.index(), closeness_centrality);
            }
        }
    }

    fn calculate_betweenness_centrality(&self) {
        println!("Betweenness Centrality:");

        for node in self.graph.node_indices() {
            // 计算介数中心度
            let betweenness_centrality = self.calculate_node_betweenness_centrality(node);
            println!("Node {}: {:.2}", node.index(), betweenness_centrality);
        }
    }

    fn calculate_node_betweenness_centrality(&self, target_node: NodeIndex) -> f64 {
        let mut total_betweenness = 0.0;

        // 在外部调用一次Dijkstra算法
        let all_shortest_paths = dijkstra(&self.graph, target_node, None, |_| 1);

        for source_node in self.graph.node_indices() {
            if source_node != target_node {
                for sink_node in self.graph.node_indices() {
                    if sink_node != source_node && sink_node != target_node {
                        // 直接使用之前计算的所有最短路径
                        let source_to_sink_paths = all_shortest_paths.get(&source_node).unwrap_or(&usize::MAX);
                        let sink_to_target_paths = all_shortest_paths.get(&sink_node).unwrap_or(&usize::MAX);

                        // 检查是否存在不可达的节点
                        if *source_to_sink_paths != usize::MAX && *sink_to_target_paths != usize::MAX {
                            let total_paths = *source_to_sink_paths;
                            let contributing_paths = *source_to_sink_paths * *sink_to_target_paths;

                            total_betweenness += contributing_paths as f64 / total_paths as f64;
                        }
                    }
                }
            }
        }

        total_betweenness
    }

}

fn main() {
    let mut graph_analysis = GraphAnalysis::new();

    // load data
    // graph_analysis.load_data();

    // Create a simple graph
    graph_analysis.create_simple_graph();

    // example：Find the similarity of friend sets between two vertices
    let vertex1 = NodeIndex::new(0);
    let vertex2 = NodeIndex::new(1);
    graph_analysis.friends_of_friends_similarity(vertex1, vertex2);

    // Example: Calculating the distance between two vertices and printing
    let start_node = NodeIndex::new(0);
    let end_node = NodeIndex::new(1);
    if let Some(distance) = graph_analysis.calculate_distance(start_node, end_node) {
        println!("Distance between {:?} and {:?}: {}", start_node.index(), end_node.index(), distance);
    } else {
        println!("Nodes are not connected.");
    }

    // Example: Find the most representative node
    let k = 2;
    graph_analysis.graph_clustering(k);


    // Example: Logic for executing the closest subgraph problem
    graph_analysis.densest_subgraph();

    // Example: Performing centrality measurement
    graph_analysis.centrality_measures();
}
