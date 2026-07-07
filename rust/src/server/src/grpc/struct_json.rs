//! Lossless-enough conversion between JSON values and protobuf `Struct`.

pub(crate) fn json_to_prost_struct(value: &serde_json::Value) -> Option<prost_types::Struct> {
    match value {
        serde_json::Value::Object(map) => Some(prost_types::Struct {
            fields: map
                .iter()
                .map(|(key, value)| (key.clone(), json_to_prost_value(value)))
                .collect(),
        }),
        _ => None,
    }
}

fn json_to_prost_value(value: &serde_json::Value) -> prost_types::Value {
    use prost_types::value::Kind;
    let kind = match value {
        serde_json::Value::Null => Kind::NullValue(prost_types::NullValue::NullValue as i32),
        serde_json::Value::Bool(value) => Kind::BoolValue(*value),
        serde_json::Value::Number(value) => Kind::NumberValue(value.as_f64().unwrap_or(0.0)),
        serde_json::Value::String(value) => Kind::StringValue(value.clone()),
        serde_json::Value::Array(values) => Kind::ListValue(prost_types::ListValue {
            values: values.iter().map(json_to_prost_value).collect(),
        }),
        serde_json::Value::Object(values) => Kind::StructValue(prost_types::Struct {
            fields: values
                .iter()
                .map(|(key, value)| (key.clone(), json_to_prost_value(value)))
                .collect(),
        }),
    };
    prost_types::Value { kind: Some(kind) }
}

pub(crate) fn prost_struct_to_json(value: &prost_types::Struct) -> serde_json::Value {
    serde_json::Value::Object(
        value
            .fields
            .iter()
            .map(|(key, value)| (key.clone(), prost_value_to_json(value)))
            .collect(),
    )
}

fn prost_value_to_json(value: &prost_types::Value) -> serde_json::Value {
    use prost_types::value::Kind;
    match &value.kind {
        None | Some(Kind::NullValue(_)) => serde_json::Value::Null,
        Some(Kind::BoolValue(value)) => serde_json::Value::Bool(*value),
        Some(Kind::NumberValue(value)) => number_to_json(*value),
        Some(Kind::StringValue(value)) => serde_json::Value::String(value.clone()),
        Some(Kind::ListValue(values)) => {
            serde_json::Value::Array(values.values.iter().map(prost_value_to_json).collect())
        }
        Some(Kind::StructValue(value)) => prost_struct_to_json(value),
    }
}

fn number_to_json(value: f64) -> serde_json::Value {
    if value.is_finite()
        && value.fract() == 0.0
        && value >= i64::MIN as f64
        && value <= i64::MAX as f64
    {
        serde_json::Value::Number((value as i64).into())
    } else {
        serde_json::Number::from_f64(value)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integral_struct_numbers_return_as_json_integers() {
        let input = serde_json::json!({"port": 8000, "ratio": 1.5});
        let round_trip = prost_struct_to_json(&json_to_prost_struct(&input).unwrap());
        assert_eq!(round_trip, input);
        assert!(round_trip["port"].is_i64());
    }
}
