"""
Clustering module for entity resolution.

This module provides the Clusterer class, which handles grouping of matched
entity pairs into clusters representing the same real-world entity.
"""

import os
import logging
import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
import networkx as nx
import community as community_louvain
from itertools import combinations

from src.utils import save_checkpoint, load_checkpoint, Timer

logger = logging.getLogger(__name__)

class Clusterer:
    """
    Handles clustering of matched entity pairs into entity clusters.
    
    Features:
    - Graph-based community detection
    - Transitivity enforcement
    - Resolution of cluster conflicts
    - Serialization of identity graph
    """
    
    def __init__(self, config):
        """
        Initialize the clusterer with configuration parameters.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config
        
        # Clustering parameters
        self.algorithm = config['clustering']['algorithm']
        self.min_edge_weight = config['clustering']['min_edge_weight']
        self.transitivity_enabled = config['clustering']['transitivity_enabled']
        self.resolve_conflicts = config['clustering']['resolve_conflicts']
        self.min_cluster_size = config['clustering']['min_cluster_size']
        
        # Initialize data structures
        self.graph = nx.Graph()
        self.clusters = []
        self.metrics = {}
        
        logger.info("Clusterer initialized with algorithm: %s", self.algorithm)

    def execute(self, checkpoint=None):
        """
        Execute clustering of matched entity pairs.
        
        Args:
            checkpoint (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Clustering results
        """
        # Load checkpoint if provided
        if checkpoint and os.path.exists(checkpoint):
            state = load_checkpoint(checkpoint)
            
            # Load graph from adjacency list
            adjacency_list = state.get('adjacency_list', {})
            self.graph = nx.Graph()
            
            for node, neighbors in adjacency_list.items():
                for neighbor, weight in neighbors.items():
                    self.graph.add_edge(node, neighbor, weight=weight)
            
            self.clusters = state.get('clusters', [])
            self.metrics = state.get('metrics', {})
            
            logger.info("Resumed clustering from checkpoint: %s", checkpoint)
        else:
            # Load match results
            matches, prefiltered = self._load_matches()
            
            # Build graph
            self._build_graph(matches, prefiltered)
        
        logger.info("Graph has %d nodes and %d edges", 
                   self.graph.number_of_nodes(), self.graph.number_of_edges())
        
        # Apply transitivity if enabled
        if self.transitivity_enabled:
            logger.info("Applying transitivity")
            self._apply_transitivity()
        
        # Perform clustering
        try:
            with Timer() as timer:
                logger.info("Performing clustering with algorithm: %s", self.algorithm)
                
                if self.algorithm == 'connected_components':
                    self.clusters = self._cluster_connected_components()
                
                elif self.algorithm == 'louvain':
                    self.clusters = self._cluster_louvain()
                
                elif self.algorithm == 'label_propagation':
                    self.clusters = self._cluster_label_propagation()
                
                else:
                    raise ValueError(f"Unsupported clustering algorithm: {self.algorithm}")
                
                # Safely access timer duration
                duration = timer.duration if hasattr(timer, 'duration') and timer.duration is not None else 0
                logger.info("Clustering completed in %.2f seconds", duration)
        except Exception as e:
            logger.error(f"Error during clustering: {str(e)}")
            # Provide fallback clusters if needed
            if not self.clusters:
                logger.warning("Using empty clusters due to clustering error")
                self.clusters = []
        
        # Filter small clusters if needed
        if self.min_cluster_size > 1:
            logger.info("Filtering clusters with fewer than %d entities", self.min_cluster_size)
            self.clusters = [c for c in self.clusters if len(c) >= self.min_cluster_size]
        
        # Resolve conflicts if enabled
        if self.resolve_conflicts:
            logger.info("Resolving cluster conflicts")
            self._resolve_conflicts()
        
        # Calculate metrics
        self._calculate_metrics()
        
        # Save results
        self._save_results()
        
        results = {
            'clusters': len(self.clusters),
            'total_entities': sum(len(c) for c in self.clusters),
            'singleton_clusters': sum(1 for c in self.clusters if len(c) == 1),
            'max_cluster_size': max(len(c) for c in self.clusters) if self.clusters else 0,
            'duration': timer.duration
        }
        
        logger.info("Clustering completed: %d clusters, %d total entities",
                   len(self.clusters), sum(len(c) for c in self.clusters))
        
        return results

    def _load_matches(self):
        """
        Load match results from classification.
        
        Returns:
            tuple: (matches, prefiltered)
        """
        try:
            output_dir = Path(self.config['system']['output_dir'])
            
            # Load classified matches
            classifier_output_path = output_dir / "classified_pairs.json"
            if not classifier_output_path.exists():
                # Placeholder for demo purposes
                matches = []
                logger.warning("No classified pairs found, using empty list")
            else:
                with open(classifier_output_path, 'r') as f:
                    matches = json.load(f)
            
            # Load prefiltered matches
            prefiltered_true_path = output_dir / "prefiltered_true.json"
            if prefiltered_true_path.exists():
                with open(prefiltered_true_path, 'r') as f:
                    prefiltered_true = json.load(f)
            else:
                prefiltered_true = []
            
            logger.info("Loaded %d matches and %d prefiltered matches",
                       len(matches), len(prefiltered_true))
            
            return matches, prefiltered_true
        
        except Exception as e:
            logger.error("Error loading matches: %s", str(e))
            return [], []

    def _build_graph(self, matches, prefiltered):
        """
        Build entity graph from match results.
        
        Args:
            matches (list): List of matched pairs
            prefiltered (list): List of prefiltered matches
        """
        # Initialize graph
        self.graph = nx.Graph()
        
        # Add matches as edges
        for match in matches:
            record1_id = match.get('record1_id')
            record2_id = match.get('record2_id')
            confidence = match.get('confidence', 0.5)
            
            if record1_id and record2_id and confidence >= self.min_edge_weight:
                self.graph.add_edge(record1_id, record2_id, weight=confidence)
        
        # Add prefiltered matches as edges
        for match in prefiltered:
            record1_id = match.get('record1_id')
            record2_id = match.get('record2_id')
            
            if record1_id and record2_id:
                self.graph.add_edge(record1_id, record2_id, weight=1.0)

    def _apply_transitivity(self):
        """
        Apply transitivity to the entity graph.
        
        If A matches B and B matches C, add an edge between A and C.
        """
        # Get a copy of the current graph
        original_graph = self.graph.copy()
        
        # Find all connected components
        components = list(nx.connected_components(original_graph))
        
        # Process each component
        for component in tqdm(components, desc="Applying transitivity"):
            # Skip small components
            if len(component) <= 2:
                continue
            
            # Get subgraph for this component
            subgraph = original_graph.subgraph(component)
            
            # Find missing edges
            nodes = list(component)
            for i, j in combinations(range(len(nodes)), 2):
                node1, node2 = nodes[i], nodes[j]
                
                # Skip if edge already exists
                if original_graph.has_edge(node1, node2):
                    continue
                
                # Check if there's a path between the nodes
                try:
                    path = nx.shortest_path(subgraph, node1, node2)
                    
                    # If path exists and is short enough, add edge
                    if 2 <= len(path) <= 3:  # Path length of 2 or 3 (1 or 2 intermediaries)
                        # Calculate weight based on path weights
                        path_edges = list(zip(path[:-1], path[1:]))
                        edge_weights = [original_graph[u][v]['weight'] for u, v in path_edges]
                        
                        # Use minimum weight along the path
                        min_weight = min(edge_weights)
                        
                        # Scale down by path length
                        transitive_weight = min_weight * 0.9  # Scale down slightly
                        
                        # Add edge if weight is above threshold
                        if transitive_weight >= self.min_edge_weight:
                            self.graph.add_edge(node1, node2, weight=transitive_weight)
                except nx.NetworkXNoPath:
                    # No path exists
                    continue

    def _cluster_connected_components(self):
        """
        Cluster entities using connected components.
        
        Returns:
            list: List of clusters, where each cluster is a list of entity IDs
        """
        # Find connected components
        components = list(nx.connected_components(self.graph))
        
        return [list(component) for component in components]

    def _cluster_louvain(self):
        """
        Cluster entities using Louvain community detection.
        
        Returns:
            list: List of clusters, where each cluster is a list of entity IDs
        """
        # Apply Louvain community detection
        partition = community_louvain.best_partition(self.graph)
        
        # Group nodes by community
        communities = {}
        for node, community_id in partition.items():
            if community_id not in communities:
                communities[community_id] = []
            
            communities[community_id].append(node)
        
        return list(communities.values())

    def _cluster_label_propagation(self):
        """
        Cluster entities using label propagation.
        
        Returns:
            list: List of clusters, where each cluster is a list of entity IDs
        """
        # Apply label propagation
        partition = nx.algorithms.community.label_propagation_communities(self.graph)
        
        return [list(community) for community in partition]

    def _resolve_conflicts(self):
        """
        Resolve conflicts between clusters.
        
        If the same entity appears in multiple clusters, assign it to the
        cluster with the strongest connections.
        """
        # Find entities that appear in multiple clusters
        entity_to_clusters = {}
        
        for i, cluster in enumerate(self.clusters):
            for entity in cluster:
                if entity not in entity_to_clusters:
                    entity_to_clusters[entity] = []
                
                entity_to_clusters[entity].append(i)
        
        # Find entities with conflicts
        conflicts = {entity: clusters for entity, clusters in entity_to_clusters.items()
                   if len(clusters) > 1}
        
        if not conflicts:
            return
        
        logger.info("Found %d entities with conflicts", len(conflicts))
        
        # Resolve conflicts
        resolved_clusters = [set(cluster) for cluster in self.clusters]
        
        for entity, cluster_indices in conflicts.items():
            # Calculate connection strength to each cluster
            strengths = []
            
            for cluster_idx in cluster_indices:
                cluster = resolved_clusters[cluster_idx]
                
                # Skip if entity is no longer in this cluster
                if entity not in cluster:
                    strengths.append(0.0)
                    continue
                
                # Calculate total edge weight to this cluster
                total_weight = sum(self.graph[entity][neighbor].get('weight', 0.0)
                                for neighbor in cluster if neighbor != entity and neighbor in self.graph[entity])
                
                # Normalize by cluster size
                strength = total_weight / (len(cluster) - 1) if len(cluster) > 1 else 0.0
                strengths.append(strength)
            
            # Find strongest cluster
            if strengths:
                strongest_idx = cluster_indices[np.argmax(strengths)]
                
                # Remove from all other clusters
                for cluster_idx in cluster_indices:
                    if cluster_idx != strongest_idx:
                        resolved_clusters[cluster_idx].discard(entity)
        
        # Update clusters
        self.clusters = [list(cluster) for cluster in resolved_clusters if cluster]

    def _calculate_metrics(self):
        """
        Calculate clustering metrics.
        """
        # Count clusters by size
        size_counts = {}
        for cluster in self.clusters:
            size = len(cluster)
            size_counts[size] = size_counts.get(size, 0) + 1
        
        # Calculate statistics
        cluster_sizes = [len(cluster) for cluster in self.clusters]
        
        self.metrics = {
            'cluster_count': len(self.clusters),
            'total_entities': sum(cluster_sizes),
            'singleton_clusters': size_counts.get(1, 0),
            'size_distribution': size_counts,
            'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'mean_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'median_cluster_size': np.median(cluster_sizes) if cluster_sizes else 0
        }

    def _save_results(self):
        """
        Save clustering results.
        """
        output_dir = Path(self.config['system']['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        # Save clusters
        with open(output_dir / "entity_clusters.json", 'w') as f:
            json.dump(self.clusters, f, indent=2)
        
        # Save metrics
        with open(output_dir / "clustering_metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save graph as adjacency list (more compact than full graph)
        adjacency_list = {}
        for node in self.graph.nodes():
            adjacency_list[node] = {}
            for neighbor in self.graph.neighbors(node):
                adjacency_list[node][neighbor] = self.graph[node][neighbor].get('weight', 1.0)
        
        # Save final checkpoint
        checkpoint_path = Path(self.config['system']['checkpoint_dir']) / "clustering_final.ckpt"
        save_checkpoint({
            'adjacency_list': adjacency_list,
            'clusters': self.clusters,
            'metrics': self.metrics
        }, checkpoint_path)
        
        # Generate cluster visualization
        self._visualize_clusters(output_dir / "cluster_visualization.png")
        
        logger.info("Clustering results saved to %s", output_dir)

    def _visualize_clusters(self, output_path):
        """
        Generate cluster visualization.
        
        Args:
            output_path (Path): Output file path
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create cluster size distribution plot
            sizes = [len(cluster) for cluster in self.clusters]
            
            plt.figure(figsize=(10, 6))
            plt.hist(sizes, bins=20)
            plt.xlabel('Cluster Size')
            plt.ylabel('Count')
            plt.title('Cluster Size Distribution')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            
            plt.savefig(output_path)
            plt.close()
        
        except Exception as e:
            logger.error("Error generating cluster visualization: %s", str(e))

    def get_cluster_for_entity(self, entity_id):
        """
        Get the cluster containing a specific entity.
        
        Args:
            entity_id (str): Entity ID
            
        Returns:
            list: Cluster containing the entity, or None if not found
        """
        for cluster in self.clusters:
            if entity_id in cluster:
                return cluster
        
        return None

    def get_entity_cluster_id(self, entity_id):
        """
        Get the cluster ID for a specific entity.
        
        Args:
            entity_id (str): Entity ID
            
        Returns:
            int: Cluster ID, or -1 if not found
        """
        for i, cluster in enumerate(self.clusters):
            if entity_id in cluster:
                return i
        
        return -1
