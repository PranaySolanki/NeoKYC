import React from 'react';
import { StyleSheet, Text, View } from 'react-native';

export const FaceLivenessScreen = () => {
    return (
        <View style={styles.container}>
            <Text style={styles.title}>Liveness Check</Text>
            <Text style={styles.subtitle}>Position your face inside the oval guide</Text>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#0F172A',
        alignItems: 'center',
        justifyContent: 'center',
    },
    title: {
        color: '#F8FAFC',
        fontSize: 22,
        fontWeight: '700',
        marginBottom: 8,
    },
    subtitle: {
        color: '#94A3B8',
        fontSize: 14,
    },
});