import logging
import weaviate
from weaviate.classes.query import Filter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_weaviate_collections():
    """
    Debug function to analyze the content of Weaviate collections.
    """
    try:
        # Connect to Weaviate
        client = weaviate.connect_to_local()
        
        # Get list of collections
        collections = client.collections.list_all()
        logger.info(f"Collections in Weaviate: {collections}")
        
        # Get the collection
        collection_name = "UniqueStringsByField"  # Update with your actual collection name
        if collection_name in collections:
            collection = client.collections.get(collection_name)
            
            # Get total count
            count_result = collection.aggregate.over_all(total_count=True)
            logger.info(f"Total objects in collection: {count_result.total_count}")
            
            # Analyze distribution by field_type
            from weaviate.classes.aggregate import GroupByAggregate
            
            field_type_result = collection.aggregate.over_all(
                group_by=GroupByAggregate(prop="field_type"),
                total_count=True
            )
            
            logger.info("Field type distribution:")
            for group in field_type_result.groups:
                logger.info(f"  {group.grouped_by.value}: {group.total_count}")
            
            # Sample a few objects to verify
            from weaviate.classes.query import QueryNearText
            
            results = collection.query.fetch_objects(limit=5, include_vector=True)
            logger.info(f"Sample objects: {len(results.objects)}")
            
            for obj in results.objects:
                logger.info(f"Object fields: {obj.properties}")
                logger.info(f"Vector dimensions: {len(next(iter(obj.vector.values()))) if obj.vector else 'No vector'}")
            
            # Close client
            client.close()
            
            return {
                "total_count": count_result.total_count,
                "field_distribution": {
                    group.grouped_by.value: group.total_count for group in field_type_result.groups
                }
            }
        else:
            logger.warning(f"Collection {collection_name} not found")
            return {}
    
    except Exception as e:
        logger.error(f"Error analyzing Weaviate collections: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}

if __name__ == "__main__":
    # Run the debug function
    stats = debug_weaviate_collections()
    print(f"Collection stats: {stats}")