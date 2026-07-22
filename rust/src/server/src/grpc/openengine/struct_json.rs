// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Loss-aware conversion between protobuf `Struct` and `serde_json::Value`.

const MAX_EXACT_JSON_INTEGER: i128 = 1_i128 << 53;

pub(super) fn prost_struct_to_json(value: &prost_types::Struct) -> serde_json::Value {
    serde_json::Value::Object(
        value
            .fields
            .iter()
            .map(|(key, value)| (key.clone(), prost_value_to_json(value)))
            .collect(),
    )
}

/// Restore exact integral protobuf numbers before handing engine-owned data
/// back to vLLM. `google.protobuf.Struct` represents every number as a double,
/// but NIXL requires block IDs, ports, ranks, and token counts to remain JSON
/// integers. Decimal strings used for integers larger than 53 bits stay opaque.
pub(super) fn prost_handoff_struct_to_json(value: &prost_types::Struct) -> serde_json::Value {
    serde_json::Value::Object(
        value
            .fields
            .iter()
            .map(|(key, value)| (key.clone(), prost_handoff_value_to_json(value)))
            .collect(),
    )
}

fn prost_value_to_json(value: &prost_types::Value) -> serde_json::Value {
    use prost_types::value::Kind;
    match value.kind.as_ref() {
        None | Some(Kind::NullValue(_)) => serde_json::Value::Null,
        Some(Kind::BoolValue(value)) => serde_json::Value::Bool(*value),
        Some(Kind::NumberValue(value)) => serde_json::json!(*value),
        Some(Kind::StringValue(value)) => serde_json::Value::String(value.clone()),
        Some(Kind::ListValue(value)) => {
            serde_json::Value::Array(value.values.iter().map(prost_value_to_json).collect())
        }
        Some(Kind::StructValue(value)) => prost_struct_to_json(value),
    }
}

fn prost_handoff_value_to_json(value: &prost_types::Value) -> serde_json::Value {
    use prost_types::value::Kind;
    match value.kind.as_ref() {
        Some(Kind::NumberValue(value))
            if value.is_finite()
                && value.fract() == 0.0
                && value.abs() <= MAX_EXACT_JSON_INTEGER as f64 =>
        {
            if *value >= 0.0 {
                serde_json::Value::Number(serde_json::Number::from(*value as u64))
            } else {
                serde_json::Value::Number(serde_json::Number::from(*value as i64))
            }
        }
        Some(Kind::ListValue(value)) => {
            serde_json::Value::Array(value.values.iter().map(prost_handoff_value_to_json).collect())
        }
        Some(Kind::StructValue(value)) => prost_handoff_struct_to_json(value),
        _ => prost_value_to_json(value),
    }
}

pub(super) fn json_to_prost_struct(value: &serde_json::Value) -> Option<prost_types::Struct> {
    let serde_json::Value::Object(fields) = value else {
        return None;
    };
    Some(prost_types::Struct {
        fields: fields
            .iter()
            .map(|(key, value)| (key.clone(), json_to_prost_value(value)))
            .collect(),
    })
}

fn json_to_prost_value(value: &serde_json::Value) -> prost_types::Value {
    use prost_types::value::Kind;
    let kind = match value {
        serde_json::Value::Null => Kind::NullValue(0),
        serde_json::Value::Bool(value) => Kind::BoolValue(*value),
        serde_json::Value::Number(value) => {
            if value
                .as_i64()
                .is_some_and(|value| i128::from(value).abs() > MAX_EXACT_JSON_INTEGER)
                || value.as_u64().is_some_and(|value| i128::from(value) > MAX_EXACT_JSON_INTEGER)
            {
                Kind::StringValue(value.to_string())
            } else {
                Kind::NumberValue(value.as_f64().unwrap_or_default())
            }
        }
        serde_json::Value::String(value) => Kind::StringValue(value.clone()),
        serde_json::Value::Array(values) => Kind::ListValue(prost_types::ListValue {
            values: values.iter().map(json_to_prost_value).collect(),
        }),
        serde_json::Value::Object(fields) => Kind::StructValue(prost_types::Struct {
            fields: fields
                .iter()
                .map(|(key, value)| (key.clone(), json_to_prost_value(value)))
                .collect(),
        }),
    };
    prost_types::Value { kind: Some(kind) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_json_integers_are_decimal_strings() {
        let input = serde_json::json!({"small": 42, "large": 9_007_199_254_740_993_u64});
        let encoded = json_to_prost_struct(&input).unwrap();
        let decoded = prost_struct_to_json(&encoded);
        assert_eq!(
            decoded,
            serde_json::json!({"small": 42.0, "large": "9007199254740993"})
        );
    }

    #[test]
    fn handoff_json_restores_exact_integral_numbers() {
        let input = serde_json::json!({
            "remote_block_ids": [[1, 2]],
            "remote_port": 14579,
            "remote_blocks_expiry_time": 12.5,
            "large": 9_007_199_254_740_993_u64,
        });
        let encoded = json_to_prost_struct(&input).unwrap();
        assert_eq!(
            prost_handoff_struct_to_json(&encoded),
            serde_json::json!({
                "remote_block_ids": [[1, 2]],
                "remote_port": 14579,
                "remote_blocks_expiry_time": 12.5,
                "large": "9007199254740993",
            })
        );
    }
}
