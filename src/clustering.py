"""
Improved clustering module for entity resolution.

This module provides the Clusterer class, which handles grouping of matched
entity pairs into clusters representing the same real-world entity with 
enhanced robustness and error handling.
"""

import os
import logging
import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
import networkx as nx
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import save_checkpoint, load_checkpoint, Timer

logger = logging.getLogger(__name__)

class Clusterer:
    """
    Handles clustering of matched entity pairs into entity clusters with improved
    robustness and error handling.
    
    Features:
    - Graph-based community detection
    - Transitivity enforcement
    - Enhanced conflict resolution
    - Better serialization and checkpointing
    - Robust error recovery
    - Detailed diagnostics
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
        
        # Additional parameters
        self.max_cluster_size = config['clustering'].get('max_cluster_size', 1000)
        self.connectivity_threshold = config['clustering'].get('connectivity_threshold', 0.5)
        self.allow_singletons = config['clustering'].get('allow_singletons', True)
        
        # Initialize data structures
        self.graph = nx.Graph()
        self.clusters = []
        self.metrics = {}
        self.entity_to_cluster = {}  # Mapping from entities to cluster IDs
        
        # Set up paths
        self.output_dir = Path(self.config['system']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = Path(self.config['system']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        self.viz_dir = self.output_dir / "visualizations" / "clustering"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Clusterer initialized with algorithm: %s", self.algorithm)

    def execute(self, checkpoint=None):
        """
        Execute clustering of matched entity pairs with enhanced robustness.
        
        Args:
            checkpoint (str, optional): Path to checkpoint file. Defaults to None.
            
        Returns:
            dict: Clustering results
        """
        # Load checkpoint if provided
        if checkpoint and os.path.exists(checkpoint):
            state = self._load_checkpoint(checkpoint)
            if state:
                logger.info("Resumed clustering from checkpoint: %s", checkpoint)
        
        # Build graph from classified pairs if not already built
        if self.graph.number_of_nodes() == 0:
            self._build_graph()
        
        logger.info("Graph has %d nodes and %d edges", 
                   self.graph.number_of_nodes(), self.graph.number_of_edges())
        
        # Save initial graph state for diagnostics
        self._save_graph_stats("initial_graph")
        
        # Apply transitivity if enabled
        if self.transitivity_enabled:
            logger.info("Applying transitivity")
            with Timer() as timer:
                added_edges = self._apply_transitivity()
                logger.info(f"Added {added_edges} transitive edges in {timer.duration:.2f} seconds")
            
            # Save post-transitivity graph state
            self._save_graph_stats("post_transitivity_graph")
        
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
                
                elif self.algorithm == 'hierarchical':
                    self.clusters = self._cluster_hierarchical()
                
                else:
                    raise ValueError(f"Unsupported clustering algorithm: {self.algorithm}")
                
                logger.info("Clustering completed in %.2f seconds with %d clusters", 
                           timer.duration, len(self.clusters))
        except Exception as e:
            logger.error(f"Error during clustering: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Fall back to connected components if primary method fails
            logger.warning("Falling back to connected components clustering")
            self.clusters = self._cluster_connected_components()
            
            if not self.clusters:
                logger.error("Failed to create clusters with fallback method")
                # Create empty cluster result
                self.clusters = []
        
        # Filter clusters by size
        logger.info("Filtering clusters based on size criteria")
        self._filter_clusters()
        
        # Resolve conflicts if enabled
        if self.resolve_conflicts:
            logger.info("Resolving cluster conflicts")
            with Timer() as timer:
                resolved = self._resolve_conflicts()
                logger.info(f"Resolved {resolved} conflicts in {timer.duration:.2f} seconds")
        
        # Build entity to cluster mapping
        self._build_entity_to_cluster_mapping()
        
        # Calculate metrics
        self._calculate_metrics()
        
        # Save results
        self._save_results()
        
        results = {
            'clusters': len(self.clusters),
            'total_entities': sum(len(c) for c in self.clusters),
            'singleton_clusters': sum(1 for c in self.clusters if len(c) == 1),
            'max_cluster_size': max((len(c) for c in self.clusters), default=0),
            'min_cluster_size': min((len(c) for c in self.clusters), default=0) if self.clusters else 0,
            'metrics': self.metrics
        }
        
        logger.info("Clustering completed: %d clusters, %d total entities",
                   len(self.clusters), sum(len(c) for c in self.clusters))
        
        return results

    def _build_graph(self):
        """
        Build entity graph from classified pairs with enhanced robustness.
        """
        logger.info("Building entity graph from classified pairs")
        
        # Initialize graph
        self.graph = nx.Graph()
        total_pairs = 0
        edges_added = 0
        
        try:
            # First try to load classified pairs
            classified_pairs = self._load_classified_pairs()
            if classified_pairs:
                logger.info(f"Loaded {len(classified_pairs)} classified pairs")
                total_pairs = len(classified_pairs)
                
                for pair in tqdm(classified_pairs, desc="Adding edges from classified pairs"):
                    try:
                        record1_id = pair.get('record1_id')
                        record2_id = pair.get('record2_id')
                        prediction = pair.get('prediction', 0)
                        confidence = pair.get('confidence', 0.5)
                        
                        if not record1_id or not record2_id:
                            continue
                        
                        # Add edge if predicted as match
                        if prediction == 1 and confidence >= self.min_edge_weight:
                            self.graph.add_edge(record1_id, record2_id, weight=confidence)
                            edges_added += 1
                    except Exception as e:
                        logger.error(f"Error processing classified pair: {e}")
                        continue
            else:
                # Fall back to pre-filtered pairs if available
                prefiltered_true = self._load_prefiltered_pairs()
                if prefiltered_true:
                    logger.info(f"Using {len(prefiltered_true)} prefiltered pairs")
                    total_pairs = len(prefiltered_true)
                    
                    for pair in tqdm(prefiltered_true, desc="Adding edges from prefiltered pairs"):
                        try:
                            record1_id = pair.get('record1_id')
                            record2_id = pair.get('record2_id')
                            
                            if not record1_id or not record2_id:
                                continue
                            
                            # Add edge with default weight
                            self.graph.add_edge(record1_id, record2_id, weight=1.0)
                            edges_added += 1
                        except Exception as e:
                            logger.error(f"Error processing prefiltered pair: {e}")
                            continue
                else:
                    logger.warning("No classification data found, graph will be empty")
        
        except Exception as e:
            logger.error(f"Error building graph: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        logger.info(f"Added {edges_added} edges from {total_pairs} pairs ({edges_added/max(1, total_pairs):.2%} conversion rate)")
        
        # Save the initial graph structure for diagnostics
        self._save_graph_visualization("initial_graph")
        
        return self.graph.number_of_edges() > 0

    def _load_classified_pairs(self):
        """
        Load classified pairs with support for multiple formats and sources.
        
        Returns:
            list: Classified pairs or empty list if not found
        """
        try:
            # Try multiple possible file names and formats
            possible_files = [
                "classified_pairs.json",
                "classifier_output.json",
                "match_pairs.json"
            ]
            
            for file_name in possible_files:
                file_path = self.output_dir / file_name
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        pairs = json.load(f)
                        logger.info(f"Loaded {len(pairs)} pairs from {file_path}")
                        return pairs
            
            # If no JSON file found, try pickle files
            pickle_path = self.output_dir / "classified_pairs.pkl"
            if pickle_path.exists():
                with open(pickle_path, 'rb') as f:
                    pairs = pickle.load(f)
                    logger.info(f"Loaded {len(pairs)} pairs from {pickle_path}")
                    return pairs
            
            logger.warning("No classified pairs file found")
            return []
        
        except Exception as e:
            logger.error(f"Error loading classified pairs: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def _load_prefiltered_pairs(self):
        """
        Load prefiltered true pairs for fallback.
        
        Returns:
            list: Prefiltered pairs or empty list if not found
        """
        try:
            # Try multiple possible file names
            possible_files = [
                "prefiltered_true.json",
                "prefiltered_matches.json"
            ]
            
            for file_name in possible_files:
                file_path = self.output_dir / file_name
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        pairs = json.load(f)
                        logger.info(f"Loaded {len(pairs)} prefiltered pairs from {file_path}")
                        return pairs
            
            # If no JSON file found, try pickle files
            pickle_path = self.output_dir / "prefiltered_true.pkl"
            if pickle_path.exists():
                with open(pickle_path, 'rb') as f:
                    pairs = pickle.load(f)
                    logger.info(f"Loaded {len(pairs)} prefiltered pairs from {pickle_path}")
                    return pairs
            
            logger.warning("No prefiltered pairs file found")
            return []
        
        except Exception as e:
            logger.error(f"Error loading prefiltered pairs: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def _apply_transitivity(self):
        """
        Apply transitivity to the entity graph with enhanced efficiency.
        
        If A matches B and B matches C, add an edge between A and C.
        
        Returns:
            int: Number of transitive edges added
        """
        # Get a copy of the current graph for reference
        original_graph = self.graph.copy()
        added_edges = 0
        
        # Process each node to find potential transitive edges
        for node in tqdm(original_graph.nodes(), desc="Applying transitivity"):
            # Get immediate neighbors
            neighbors = set(original_graph.neighbors(node))
            
            # For each pair of neighbors, check if they should be connected
            for neighbor1 in neighbors:
                for neighbor2 in neighbors:
                    # Skip if same node or already connected
                    if neighbor1 >= neighbor2 or original_graph.has_edge(neighbor1, neighbor2):
                        continue
                    
                    # Calculate transitive edge weight as product of existing edge weights
                    weight1 = original_graph[node][neighbor1]['weight']
                    weight2 = original_graph[node][neighbor2]['weight']
                    transitive_weight = weight1 * weight2  # Product of weights
                    
                    # Add edge if weight is above threshold
                    if transitive_weight >= self.min_edge_weight:
                        self.graph.add_edge(neighbor1, neighbor2, weight=transitive_weight)
                        added_edges += 1
        
        logger.info(f"Added {added_edges} transitive edges")
        return added_edges

    def _cluster_connected_components(self):
        """
        Cluster entities using connected components.
        
        Returns:
            list: List of clusters, where each cluster is a list of entity IDs
        """
        # Find connected components
        components = list(nx.connected_components(self.graph))
        logger.info(f"Found {len(components)} connected components")
        
        # Convert components to lists for JSON serialization
        clusters = [list(component) for component in components]
        
        return clusters

    def _cluster_louvain(self):
        """
        Cluster entities using Louvain community detection.
        
        Returns:
            list: List of clusters, where each cluster is a list of entity IDs
        """
        try:
            # Import community detection library with error handling
            try:
                import community.community_louvain as community_louvain
            except ImportError:
                try:
                    import community as community_louvain
                except ImportError:
                    logger.error("Could not import community detection library")
                    # Fall back to connected components
                    logger.warning("Falling back to connected components")
                    return self._cluster_connected_components()
            
            # Apply Louvain community detection
            partition = community_louvain.best_partition(self.graph)
            
            # Group nodes by community
            communities = defaultdict(list)
            for node, community_id in partition.items():
                communities[community_id].append(node)
            
            logger.info(f"Found {len(communities)} communities using Louvain")
            
            # Convert to list of lists
            clusters = list(communities.values())
            
            return clusters
        
        except Exception as e:
            logger.error(f"Error in Louvain clustering: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Fall back to connected components
            logger.warning("Falling back to connected components")
            return self._cluster_connected_components()

    def _cluster_label_propagation(self):
        """
        Cluster entities using label propagation.
        
        Returns:
            list: List of clusters, where each cluster is a list of entity IDs
        """
        try:
            # Apply label propagation with error handling
            partition = nx.algorithms.community.label_propagation_communities(self.graph)
            
            # Convert to list of lists
            clusters = [list(community) for community in partition]
            
            logger.info(f"Found {len(clusters)} communities using label propagation")
            
            return clusters
        
        except Exception as e:
            logger.error(f"Error in label propagation clustering: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Fall back to connected components
            logger.warning("Falling back to connected components")
            return self._cluster_connected_components()

    def _cluster_hierarchical(self):
        """
        Cluster entities using hierarchical clustering.
        
        Returns:
            list: List of clusters, where each cluster is a list of entity IDs
        """
        try:
            # Process connected components separately for efficiency
            components = list(nx.connected_components(self.graph))
            logger.info(f"Processing {len(components)} connected components for hierarchical clustering")
            
            all_clusters = []
            
            # Process each component separately to avoid memory issues
            for i, component in enumerate(tqdm(components, desc="Processing components")):
                # Skip tiny components (<=2 nodes)
                if len(component) <= 2:
                    all_clusters.append(list(component))
                    continue
                
                # Skip large components that would cause memory issues
                if len(component) > 1000:
                    logger.warning(f"Component {i} has {len(component)} nodes, too large for hierarchical clustering")
                    
                    # Try to break it down using a simpler method
                    subgraph = self.graph.subgraph(component).copy()
                    
                    try:
                        # Try Louvain on this component
                        import community.community_louvain as community_louvain
                        sub_partition = community_louvain.best_partition(subgraph)
                        
                        # Group nodes by community
                        sub_communities = defaultdict(list)
                        for node, community_id in sub_partition.items():
                            sub_communities[community_id].append(node)
                        
                        # Add these communities to our clusters
                        all_clusters.extend(list(sub_communities.values()))
                    except:
                        # If Louvain fails, just add the whole component
                        all_clusters.append(list(component))
                    
                    continue
                
                # Get the subgraph for this component
                subgraph = self.graph.subgraph(component).copy()
                
                # Convert to distance matrix (1 - weight)
                nodes = list(subgraph.nodes())
                n = len(nodes)
                distance_matrix = np.ones((n, n))
                
                for i in range(n):
                    for j in range(i+1, n):
                        node1, node2 = nodes[i], nodes[j]
                        if subgraph.has_edge(node1, node2):
                            weight = subgraph[node1][node2]['weight']
                            distance = 1.0 - weight
                            distance_matrix[i, j] = distance
                            distance_matrix[j, i] = distance
                
                # Perform hierarchical clustering
                from scipy.cluster.hierarchy import linkage, fcluster
                Z = linkage(distance_matrix[np.triu_indices(n, k=1)], method='average')
                
                # Cut the dendrogram to form clusters
                threshold = 1.0 - self.connectivity_threshold
                cluster_labels = fcluster(Z, threshold, criterion='distance')
                
                # Group nodes by cluster label
                component_clusters = defaultdict(list)
                for i, label in enumerate(cluster_labels):
                    component_clusters[label].append(nodes[i])
                
                # Add clusters from this component
                all_clusters.extend(list(component_clusters.values()))
            
            logger.info(f"Found {len(all_clusters)} clusters using hierarchical clustering")
            
            return all_clusters
        
        except Exception as e:
            logger.error(f"Error in hierarchical clustering: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Fall back to connected components
            logger.warning("Falling back to connected components")
            return self._cluster_connected_components()

    def _filter_clusters(self):
        """
        Filter clusters based on size and quality criteria.
        """
        original_count = len(self.clusters)
        
        # Filter by minimum size if specified and singleton clusters not allowed
        if self.min_cluster_size > 1 or not self.allow_singletons:
            min_size = max(self.min_cluster_size, 2 if not self.allow_singletons else 1)
            self.clusters = [c for c in self.clusters if len(c) >= min_size]
        
        # Filter by maximum size if specified
        if self.max_cluster_size < float('inf'):
            # For large clusters, try to break them down further
            large_clusters = [c for c in self.clusters if len(c) > self.max_cluster_size]
            normal_clusters = [c for c in self.clusters if len(c) <= self.max_cluster_size]
            
            broken_clusters = []
            for large_cluster in large_clusters:
                # Get subgraph for this cluster
                subgraph = self.graph.subgraph(large_cluster).copy()
                
                # Try to break it down using a more aggressive method
                try:
                    # Remove weak edges
                    weak_edges = [(u, v) for u, v, d in subgraph.edges(data=True) 
                                 if d['weight'] < self.connectivity_threshold]
                    subgraph.remove_edges_from(weak_edges)
                    
                    # Find connected components in the reduced graph
                    sub_components = list(nx.connected_components(subgraph))
                    
                    # If we successfully broke it down, add the new smaller clusters
                    if len(sub_components) > 1:
                        broken_clusters.extend([list(c) for c in sub_components])
                    else:
                        # Could not break down, keep the original but log a warning
                        logger.warning(f"Could not break down large cluster of size {len(large_cluster)}")
                        broken_clusters.append(large_cluster)
                except Exception as e:
                    logger.error(f"Error breaking down large cluster: {e}")
                    # Keep the original cluster
                    broken_clusters.append(large_cluster)
            
            # Combine normal and broken-down clusters
            self.clusters = normal_clusters + broken_clusters
        
        filtered_count = len(self.clusters)
        logger.info(f"Filtered clusters: {original_count} -> {filtered_count}")

    def _resolve_conflicts(self):
        """
        Resolve conflicts between clusters with improved algorithm.
        
        If the same entity appears in multiple clusters, assign it to the
        cluster with the strongest connections.
        
        Returns:
            int: Number of conflicts resolved
        """
        # Count entities in each cluster
        entity_to_clusters = defaultdict(list)
        
        for i, cluster in enumerate(self.clusters):
            for entity in cluster:
                entity_to_clusters[entity].append(i)
        
        # Find entities with conflicts
        conflicts = {entity: clusters for entity, clusters in entity_to_clusters.items()
                   if len(clusters) > 1}
        
        if not conflicts:
            logger.info("No conflicts found")
            return 0
        
        logger.info(f"Found {len(conflicts)} entities with conflicts")
        
        # Resolve conflicts
        resolved_clusters = [set(cluster) for cluster in self.clusters]
        resolved_count = 0
        
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
                total_weight = 0.0
                connections = 0
                
                for neighbor in cluster:
                    if neighbor != entity and self.graph.has_edge(entity, neighbor):
                        total_weight += self.graph[entity][neighbor].get('weight', 0.0)
                        connections += 1
                
                # Normalize by cluster size
                if connections > 0:
                    strength = total_weight / connections
                else:
                    strength = 0.0
                
                strengths.append(strength)
            
            # Find strongest cluster
            if strengths:
                strongest_idx = cluster_indices[np.argmax(strengths)]
                
                # Remove from all other clusters
                for cluster_idx in cluster_indices:
                    if cluster_idx != strongest_idx:
                        resolved_clusters[cluster_idx].discard(entity)
                        resolved_count += 1
        
        # Update clusters
        self.clusters = [list(cluster) for cluster in resolved_clusters if cluster]
        
        return resolved_count

    def _build_entity_to_cluster_mapping(self):
        """
        Build mapping from entities to cluster IDs for efficient lookup.
        """
        self.entity_to_cluster = {}
        
        for cluster_id, cluster in enumerate(self.clusters):
            for entity in cluster:
                self.entity_to_cluster[entity] = cluster_id
        
        logger.info(f"Built entity-to-cluster mapping for {len(self.entity_to_cluster)} entities")

    def _calculate_metrics(self):
        """
        Calculate clustering metrics with more comprehensive statistics.
        """
        # Count clusters by size
        size_counts = {}
        for cluster in self.clusters:
            size = len(cluster)
            size_counts[size] = size_counts.get(size, 0) + 1
        
        # Calculate statistics
        cluster_sizes = [len(cluster) for cluster in self.clusters]
        
        if cluster_sizes:
            self.metrics = {
                'cluster_count': len(self.clusters),
                'total_entities': sum(cluster_sizes),
                'singleton_clusters': size_counts.get(1, 0),
                'size_distribution': size_counts,
                'min_cluster_size': min(cluster_sizes),
                'max_cluster_size': max(cluster_sizes),
                'mean_cluster_size': float(np.mean(cluster_sizes)),
                'median_cluster_size': float(np.median(cluster_sizes)),
                'std_cluster_size': float(np.std(cluster_sizes))
            }
            
            # Calculate more advanced metrics
            
            # 1. Size distribution quantiles
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
            for q in quantiles:
                self.metrics[f'size_quantile_{int(q*100)}'] = float(np.quantile(cluster_sizes, q))
            
            # 2. Graph-based metrics
            self.metrics['graph_metrics'] = {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges(),
                'density': nx.density(self.graph),
                'connected_components': nx.number_connected_components(self.graph)
            }
            
            # 3. Entity coverage
            self.metrics['entity_coverage'] = len(self.entity_to_cluster) / max(1, self.graph.number_of_nodes())
            
            # 4. Cluster size grouping (for visualization)
            size_groups = {
                '1': sum(count for size, count in size_counts.items() if size == 1),
                '2-5': sum(count for size, count in size_counts.items() if 2 <= size <= 5),
                '6-10': sum(count for size, count in size_counts.items() if 6 <= size <= 10),
                '11-20': sum(count for size, count in size_counts.items() if 11 <= size <= 20),
                '21-50': sum(count for size, count in size_counts.items() if 21 <= size <= 50),
                '51+': sum(count for size, count in size_counts.items() if size > 50)
            }
            
            self.metrics['size_groups'] = size_groups
        else:
            # Empty clusters
            self.metrics = {
                'cluster_count': 0,
                'total_entities': 0,
                'singleton_clusters': 0,
                'size_distribution': {},
                'min_cluster_size': 0,
                'max_cluster_size': 0,
                'mean_cluster_size': 0,
                'median_cluster_size': 0,
                'std_cluster_size': 0,
                'graph_metrics': {
                    'nodes': self.graph.number_of_nodes(),
                    'edges': self.graph.number_of_edges(),
                    'density': nx.density(self.graph),
                    'connected_components': nx.number_connected_components(self.graph)
                },
                'entity_coverage': 0,
                'size_groups': {
                    '1': 0, '2-5': 0, '6-10': 0, '11-20': 0, '21-50': 0, '51+': 0
                }
            }

    def _load_checkpoint(self, checkpoint_path):
        """
        Load checkpoint with better error handling.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
            
        Returns:
            bool: True if checkpoint was loaded successfully, False otherwise
        """
        try:
            state = load_checkpoint(checkpoint_path)
            
            # Check if checkpoint contains valid data
            if not state:
                logger.error("Empty or invalid checkpoint")
                return False
            
            # Try to load graph from adjacency list
            if 'adjacency_list' in state:
                adjacency_list = state['adjacency_list']
                self.graph = nx.Graph()
                
                for node, neighbors in adjacency_list.items():
                    for neighbor, weight in neighbors.items():
                        self.graph.add_edge(node, neighbor, weight=weight)
                
                logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            
            # Load clusters
            if 'clusters' in state:
                self.clusters = state['clusters']
                logger.info(f"Loaded {len(self.clusters)} clusters")
            
            # Load metrics
            if 'metrics' in state:
                self.metrics = state['metrics']
            
            # Rebuild entity to cluster mapping
            self._build_entity_to_cluster_mapping()
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _save_results(self):
        """
        Save clustering results with improved organization and format.
        """
        try:
            output_dir = self.output_dir
            
            # Save clusters
            with open(output_dir / "entity_clusters.json", 'w') as f:
                json.dump(self.clusters, f, indent=2)
            
            # Save a sample of the largest clusters
            largest_clusters = sorted(self.clusters, key=len, reverse=True)[:20]
            with open(output_dir / "largest_clusters_sample.json", 'w') as f:
                json.dump(largest_clusters, f, indent=2)
            
            # Save metrics
            with open(output_dir / "clustering_metrics.json", 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            # Save entity to cluster mapping
            with open(output_dir / "entity_to_cluster.json", 'w') as f:
                json.dump(self.entity_to_cluster, f, indent=2)
            
            # Save graph as adjacency list (more compact than full graph)
            adjacency_list = self._get_adjacency_list()
            
            # Save final checkpoint
            checkpoint_path = self.checkpoint_dir / "clustering_final.ckpt"
            save_checkpoint({
                'adjacency_list': adjacency_list,
                'clusters': self.clusters,
                'metrics': self.metrics
            }, checkpoint_path)
            
            # Generate visualizations
            self._generate_visualizations()
            
            logger.info(f"Clustering results saved to {output_dir}")
        
        except Exception as e:
            logger.error(f"Error saving clustering results: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _get_adjacency_list(self):
        """
        Get adjacency list representation of the graph.
        
        Returns:
            dict: Adjacency list
        """
        adjacency_list = {}
        
        for node in self.graph.nodes():
            adjacency_list[node] = {}
            for neighbor in self.graph.neighbors(node):
                adjacency_list[node][neighbor] = self.graph[node][neighbor].get('weight', 1.0)
        
        return adjacency_list

    def _save_graph_stats(self, prefix):
        """
        Save graph statistics for diagnostics.
        
        Args:
            prefix (str): Prefix for output filenames
        """
        try:
            # Calculate graph statistics
            graph_stats = {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges(),
                'density': nx.density(self.graph),
                'connected_components': nx.number_connected_components(self.graph),
                'average_clustering': nx.average_clustering(self.graph),
                'average_degree': sum(dict(self.graph.degree()).values()) / max(1, self.graph.number_of_nodes())
            }
            
            # Calculate degree distribution
            degree_dist = {}
            for node, degree in self.graph.degree():
                degree_dist[degree] = degree_dist.get(degree, 0) + 1
            
            graph_stats['degree_distribution'] = degree_dist
            
            # Calculate edge weight distribution
            weight_dist = {}
            for u, v, weight in self.graph.edges(data='weight', default=1.0):
                weight_bin = round(weight * 10) / 10  # Round to nearest 0.1
                weight_dist[weight_bin] = weight_dist.get(weight_bin, 0) + 1
            
            graph_stats['weight_distribution'] = weight_dist
            
            # Save statistics
            with open(self.output_dir / f"{prefix}_stats.json", 'w') as f:
                json.dump(graph_stats, f, indent=2)
            
            logger.info(f"Saved graph statistics for {prefix}")
        
        except Exception as e:
            logger.error(f"Error saving graph statistics: {e}")

    def _save_graph_visualization(self, prefix):
        """
        Save a visualization of the graph structure.
        
        Args:
            prefix (str): Prefix for output filename
        """
        try:
            # Check if graph is too large to visualize
            if self.graph.number_of_nodes() > 1000:
                logger.warning(f"Graph too large to visualize: {self.graph.number_of_nodes()} nodes")
                return
            
            # Create visualization
            plt.figure(figsize=(12, 12))
            
            # Use spring layout with seed for reproducibility
            pos = nx.spring_layout(self.graph, seed=42)
            
            # Get edge weights for width and color
            edge_weights = [self.graph[u][v].get('weight', 1.0) for u, v in self.graph.edges()]
            
            # Draw the graph
            nx.draw_networkx(
                self.graph,
                pos=pos,
                with_labels=False,
                node_size=20,
                node_color='skyblue',
                edge_color=edge_weights,
                width=[w * 2 for w in edge_weights],
                edge_cmap=plt.cm.Blues,
                alpha=0.7
            )
            
            plt.title(f"Entity Graph ({self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges)")
            plt.axis('off')
            
            # Save visualization
            plt.savefig(self.viz_dir / f"{prefix}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved graph visualization to {self.viz_dir / f'{prefix}.png'}")
        
        except Exception as e:
            logger.error(f"Error saving graph visualization: {e}")

    def _generate_visualizations(self):
        """
        Generate visualizations for clustering results.
        """
        try:
            # 1. Cluster size distribution
            self._plot_cluster_size_distribution()
            
            # 2. Degree distribution
            self._plot_degree_distribution()
            
            # 3. Edge weight distribution
            self._plot_edge_weight_distribution()
            
            # 4. Cluster size groups
            self._plot_cluster_size_groups()
            
            # 5. Sample large clusters visualization
            self._plot_sample_clusters()
            
            logger.info(f"Generated clustering visualizations in {self.viz_dir}")
        
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def _plot_cluster_size_distribution(self):
        """
        Generate cluster size distribution plot.
        """
        try:
            # Get cluster sizes
            sizes = [len(cluster) for cluster in self.clusters]
            
            if not sizes:
                logger.warning("No clusters to plot size distribution")
                return
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.hist(sizes, bins=30, log=True)
            plt.xlabel('Cluster Size')
            plt.ylabel('Number of Clusters (log scale)')
            plt.title('Cluster Size Distribution')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(self.viz_dir / "cluster_size_distribution.png")
            plt.close()
            
            logger.info("Generated cluster size distribution plot")
        
        except Exception as e:
            logger.error(f"Error generating cluster size distribution plot: {e}")

    def _plot_degree_distribution(self):
        """
        Generate node degree distribution plot.
        """
        try:
            # Get degrees
            degrees = [d for n, d in self.graph.degree()]
            
            if not degrees:
                logger.warning("No degrees to plot distribution")
                return
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.hist(degrees, bins=30, log=True)
            plt.xlabel('Node Degree')
            plt.ylabel('Number of Nodes (log scale)')
            plt.title('Node Degree Distribution')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(self.viz_dir / "degree_distribution.png")
            plt.close()
            
            logger.info("Generated node degree distribution plot")
        
        except Exception as e:
            logger.error(f"Error generating degree distribution plot: {e}")

    def _plot_edge_weight_distribution(self):
        """
        Generate edge weight distribution plot.
        """
        try:
            # Get edge weights
            weights = [d['weight'] for u, v, d in self.graph.edges(data=True) if 'weight' in d]
            
            if not weights:
                logger.warning("No edge weights to plot distribution")
                return
            
            # Create plot
            plt.figure(figsize=(10, 6))
            plt.hist(weights, bins=20)
            plt.xlabel('Edge Weight')
            plt.ylabel('Number of Edges')
            plt.title('Edge Weight Distribution')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            plt.savefig(self.viz_dir / "edge_weight_distribution.png")
            plt.close()
            
            logger.info("Generated edge weight distribution plot")
        
        except Exception as e:
            logger.error(f"Error generating edge weight distribution plot: {e}")

    def _plot_cluster_size_groups(self):
        """
        Generate cluster size groups plot.
        """
        try:
            # Get size groups from metrics
            size_groups = self.metrics.get('size_groups', {})
            
            if not size_groups:
                logger.warning("No size groups to plot")
                return
            
            # Create plot
            plt.figure(figsize=(10, 6))
            
            # Sort groups by size range
            sorted_groups = sorted(size_groups.items(), key=lambda x: x[0])
            labels = [g[0] for g in sorted_groups]
            values = [g[1] for g in sorted_groups]
            
            plt.bar(labels, values)
            plt.xlabel('Cluster Size Range')
            plt.ylabel('Number of Clusters')
            plt.title('Clusters by Size Range')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            # Save plot
            plt.savefig(self.viz_dir / "cluster_size_groups.png")
            plt.close()
            
            logger.info("Generated cluster size groups plot")
        
        except Exception as e:
            logger.error(f"Error generating cluster size groups plot: {e}")

    def _plot_sample_clusters(self):
        """
        Generate visualization of sample large clusters.
        """
        try:
            # Get largest clusters
            largest_clusters = sorted(self.clusters, key=len, reverse=True)[:5]
            
            if not largest_clusters:
                logger.warning("No clusters to visualize")
                return
            
            # Plot each large cluster
            for i, cluster in enumerate(largest_clusters):
                try:
                    if len(cluster) > 100:
                        # Skip very large clusters
                        logger.info(f"Skipping visualization of cluster {i} (size {len(cluster)})")
                        continue
                    
                    # Get subgraph for this cluster
                    subgraph = self.graph.subgraph(cluster).copy()
                    
                    # Create plot
                    plt.figure(figsize=(10, 8))
                    
                    # Use spring layout with seed for reproducibility
                    pos = nx.spring_layout(subgraph, seed=42)
                    
                    # Get edge weights for width
                    edge_weights = [subgraph[u][v].get('weight', 1.0) for u, v in subgraph.edges()]
                    
                    # Draw the subgraph
                    nx.draw_networkx(
                        subgraph,
                        pos=pos,
                        with_labels=False,
                        node_size=30,
                        node_color='skyblue',
                        edge_color=edge_weights,
                        width=[w * 2 for w in edge_weights],
                        edge_cmap=plt.cm.Blues,
                        alpha=0.7
                    )
                    
                    plt.title(f"Cluster {i+1} (size {len(cluster)})")
                    plt.axis('off')
                    
                    # Save visualization
                    plt.savefig(self.viz_dir / f"cluster_{i+1}.png", dpi=300, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    logger.error(f"Error visualizing cluster {i}: {e}")
                    continue
            
            logger.info("Generated sample cluster visualizations")
        
        except Exception as e:
            logger.error(f"Error generating sample cluster visualizations: {e}")

    def get_cluster_for_entity(self, entity_id):
        """
        Get the cluster containing a specific entity.
        
        Args:
            entity_id (str): Entity ID
            
        Returns:
            list: Cluster containing the entity, or None if not found
        """
        if entity_id in self.entity_to_cluster:
            cluster_id = self.entity_to_cluster[entity_id]
            return self.clusters[cluster_id]
        
        # Search all clusters if mapping is not available
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
        return self.entity_to_cluster.get(entity_id, -1)

    def get_largest_clusters(self, n=10):
        """
        Get the n largest clusters.
        
        Args:
            n (int): Number of clusters to return
            
        Returns:
            list: List of (cluster_id, cluster) tuples
        """
        largest = []
        
        for i, cluster in enumerate(self.clusters):
            largest.append((i, cluster))
        
        # Sort by cluster size in descending order
        largest.sort(key=lambda x: len(x[1]), reverse=True)
        
        return largest[:n]

    def export_clusters_csv(self, output_path=None):
        """
        Export clusters to CSV format.
        
        Args:
            output_path (str, optional): Output file path. Defaults to None.
            
        Returns:
            str: Path to CSV file
        """
        if output_path is None:
            output_path = self.output_dir / "entity_clusters.csv"
        
        try:
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['cluster_id', 'entity_id'])
                
                for cluster_id, cluster in enumerate(self.clusters):
                    for entity_id in cluster:
                        writer.writerow([cluster_id, entity_id])
            
            logger.info(f"Exported clusters to CSV: {output_path}")
            return str(output_path)
        
        except Exception as e:
            logger.error(f"Error exporting clusters to CSV: {e}")
            return None
