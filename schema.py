RELATION_EXTRACTION_WITH_RATIONALE_SCHEMA = {
    'name': 'relation_extraction_with_rationale',
    'parameters': {
        'type': 'object',
        'properties': {
            'triplets': {
                'type': 'array',
                'items': {
                    'type': 'string'
                },
                'description': 'The triplets extracted from the input text.'
            },
            'rationales': {
                'type':
                'array',
                'items': {
                    'type': 'string'
                },
                'description':
                'The rationales extracted from the input text. The rationales are the explanations of the triplets.'
            }
        },
        'required': ['triplets', 'rationales']
    }
}
